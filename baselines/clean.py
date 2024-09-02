import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = "cuda:2"
actor="continuous"
torch.autograd.set_detect_anomaly(True)
all_expert_states = None
all_expert_actions = None
Normalization = True
std=None
mean=None
BCO_reward_list=[]
########################################################################################
# Eval 
########################################################################################
def evaluate_real_return(policy, env, n_episodes, horizon, deterministic):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0

        for t in range(horizon):
            if Normalization:
                obs = (obs-mean)/std
            state=torch.FloatTensor(obs).to(policy.device)

            action = policy(state, deterministic, with_logprob=False)
            action = (action[0]).cpu().detach().numpy()
            # print(action.shape)
            if len(action.shape) == 3:
                 action = action.squeeze(0)
            if len(action.shape) == 2:
                action = action.squeeze(0)
            obs, rew, done, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            ret += rew 
            if done:
                break
        returns.append(ret)

    return np.mean(returns)

########################################################################################
# Model definition
########################################################################################
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU, num_bins=20):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation).to(device)
        self.logits_net = nn.Linear(hidden_sizes[-1], act_dim * num_bins).to(device)
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.bin_width = 2.0 / num_bins  # Range is [-1, 1]
        self.bin_centers = torch.linspace(-1 + self.bin_width/2, 1 - self.bin_width/2, num_bins).to(device)
        self.device = device

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        logits = self.logits_net(net_out).view(-1, self.act_dim, self.num_bins)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            actions = self.bin_centers[torch.argmax(probs, dim=-1)]
        else:
            sampled_bins = torch.multinomial(probs.view(-1, self.num_bins), 1).view(-1, self.act_dim)
            actions = self.bin_centers[sampled_bins]

        if with_logprob:
            log_prob = self.log_prob(obs, actions)
        else:
            log_prob = None
        # squeeze the action dimension

        return actions, log_prob

    def log_prob(self, obs, actions):
        net_out = self.net(obs)
        logits = self.logits_net(net_out).view(-1, self.act_dim, self.num_bins)
        probs = F.softmax(logits, dim=-1)

        # Find which bin each action falls into
        bin_indices = torch.bucketize(actions, self.bin_centers)
        
        # Clamp indices to valid range
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # Get probabilities for the selected bins
        selected_probs = probs.gather(-1, bin_indices.unsqueeze(-1)).squeeze(-1)

        # Compute log probabilities
        log_prob = torch.log(selected_probs + 1e-8)

        # Adjust for continuous approximation
        log_prob -= torch.log(torch.tensor(self.bin_width))

        return log_prob.sum(-1, keepdim=True)

    def sample(self, obs):
        net_out = self.net(obs)
        logits = self.logits_net(net_out).view(-1, self.act_dim, self.num_bins)
        probs = F.softmax(logits, dim=-1)
        sampled_bins = torch.multinomial(probs.view(-1, self.num_bins), 1).view(-1, self.act_dim)
        return self.bin_centers[sampled_bins]

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim,hidden_sizes, activation=nn.ReLU, act_limit=1):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.device = device

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def log_prob(self, obs, act):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        act = act / self.act_limit
        # act = torch.atanh(act) # arctanh to project [-1,1] to real
        # act=torch.clamp(act,max=1-1e-3,min=-1+1e-3)
        act=torch.atanh(torch.Tensor(act).to(obs.device))

        logp_pi = pi_distribution.log_prob(act).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(axis=1)



        return logp_pi
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256), activation=nn.ReLU, act_limit=1,weight_decay=False,Actor_type="continuous"):
        super().__init__()
        if Actor_type=="continuous":
            self.ac=Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif Actor_type=="discrete":
            self.ac=DiscreteActor(obs_dim, act_dim, hidden_sizes, activation)

        # self.ac=Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        # self.ac=DiscreteActor(obs_dim, act_dim, hidden_sizes, activation)
        if weight_decay:
            self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=3e-4, weight_decay=1e-5)
            # use AdamW optimizer
            # self.pi_optimizer = torch.optim.AdamW(self.ac.parameters(), lr=3e-4, weight_decay=1e-4)
        else:
            self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=3e-4)
        self.device = device
class ReplayBuffer_for_Reference_Free_Methods :
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.actions[indices]

    def __len__(self):
        return self.size
########################################################################################
# Reference-based Methods
########################################################################################
def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two multivariate Gaussian distributions.
    Assumes diagonal covariance matrices.
    """
    var1 = sigma1.pow(2)
    var2 = sigma2.pow(2)
    kl = (var1 / var2 + (mu2 - mu1).pow(2) / var2 - 1 + var2.log() - var1.log()).sum(-1) * 0.5
    return kl.mean()
def discrete_kl_divergence(logits1, logits2):
    """
    Compute KL divergence between two discrete distributions.
    """
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    kl = (p1 * (F.log_softmax(logits1, dim=-1) - F.log_softmax(logits2, dim=-1))).sum(-1)
    return kl.mean()
def DPO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6,label_smoothing=0):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            # reverse kl(original DPO)
            losses = (-F.logsigmoid(beta * logits)*(1-label_smoothing)-F.logsigmoid(-beta * logits)*(label_smoothing))


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def DPO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6,label_smoothing=0,Lambda=50):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            # reverse kl(original DPO)
            losses = (-F.logsigmoid(beta * logits)*(1-label_smoothing)-F.logsigmoid(-beta * logits)*(label_smoothing))-Lambda(max(0,reference_chosen_logps-policy_chosen_logps))


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def robustDPO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6,label_smoothing=0.1):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            # reverse kl(original DPO)
            losses = (-F.logsigmoid(beta * logits)*(1-label_smoothing)+F.logsigmoid(-beta * logits)*(label_smoothing))/(1-2*label_smoothing)


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def EXO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6,label_smoothing=1e-3):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            
            losses = (beta * logits).sigmoid()*(F.logsigmoid(beta * logits)-torch.log(1-label_smoothing))+(-beta * logits).sigmoid()*(F.logsigmoid(-beta * logits)-torch.log(label_smoothing))


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def IPO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            losses=(logits-1/(2*beta))**2


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def APOzero(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            

            losses = 1-torch.nn.functional.sigmoid(beta * chosen_logratios)+torch.nn.functional.sigmoid(beta * reject_logratios)


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def APOdown(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            

            losses = torch.nn.functional.sigmoid(beta * chosen_logratios)+1-torch.nn.functional.logsigmoid(beta * (chosen_logratios-reject_logratios))


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def BCO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3
    global BCO_reward_list

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            chosen_reward = beta * chosen_logratios
            reject_reward = beta * reject_logratios
            rewards=torch.cat((chosen_reward,reject_reward),0).mean().detach()
            BCO_reward_list.append(rewards.item())
            delta=np.mean(BCO_reward_list)
            losses = -F.logsigmoid(beta*chosen_logratios-delta)-F.logsigmoid(-beta*reject_logratios+delta)
            loss = losses.mean()
            total_loss += loss.sum()
            total_negative_reward += reject_logratios.mean().item()
            total_positive_reward += chosen_logratios.mean().item()
            total_margin += (chosen_logratios-reject_logratios).mean().item()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def KTO(agent, prev_model, expert_states, expert_actions,greedy=False, steps=100, beta=0.1, reject_from="random", clip_grad=False, noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    total_KL = 0    
    desirable_weight = 1.0  # You can adjust this weight
    undesirable_weight = 1.0  # You can adjust this weight
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            
            # Clamp the reject action to the action space
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():
                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)
            # Calculate KL divergence
            # net_out = agent.ac.net(state)
            # policy_mu = agent.ac.mu_layer(net_out)
            # policy_log_std = torch.clamp(agent.ac.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            # policy_std = torch.exp(policy_log_std)

            # with torch.no_grad():
                # prev_net_out = prev_model.net(state)
                # prev_mu = prev_model.mu_layer(prev_net_out)
                # prev_log_std = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
                # prev_std = torch.exp(prev_log_std)

            # calculate kl divergence on all expert states and actions
            # net_out = agent.ac.net(all_expert_states)
            # policy_mu = agent.ac.mu_layer(net_out)
            # policy_log_std = torch.clamp(agent.ac.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            # policy_std = torch.exp(policy_log_std)

            # with torch.no_grad():
            #     prev_net_out = prev_model.net(all_expert_states)
            #     prev_mu = prev_model.mu_layer(prev_net_out)
            #     prev_log_std = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
            #     prev_std = torch.exp(prev_log_std)
        
            # kl_divergence = gaussian_kl_divergence(policy_mu, policy_std, prev_mu, prev_std)
            chosen_kl=(policy_chosen_logps-reference_chosen_logps).mean().clamp(min=0)
            rejected_kl=(policy_rejected_logps-reference_rejected_logps).mean().clamp(min=0)
            # total_KL += kl_divergence.mean().item()


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps

            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward

            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            # chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence))
            # rejected_losses = 1 - torch.sigmoid(beta * (kl_divergence - reject_logratios))
            chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios -rejected_kl))
            rejected_losses = 1 - torch.sigmoid(beta * (chosen_kl - reject_logratios))

            losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
            loss = losses.mean()

            total_loss += loss.item()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss
    # print(total_KL)
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def SPPO(agent, prev_model, expert_states, expert_actions, greedy=False, steps=100, eta=1e3, reject_from="random", clip_grad=False, noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)           



            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = (chosen_logratios-eta/2).pow(2).mean()
            negative_reward = (reject_logratios+eta/2).pow(2).mean()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward.item()
            total_negative_reward += negative_reward.item()
            total_margin += margin


            # SPPO loss
            losses = positive_reward+ negative_reward

            loss = losses
            total_loss += loss

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.item()
    total_margin=total_margin.item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward


def AOT(agent, prev_model, expert_states, expert_actions, greedy=False, steps=100, beta=0.1, reject_from="random", clip_grad=False, noise_level=0.6, AOT_loss="logistic", sort_type="hard_sort", label_smoothing=0):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():
                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)

            chosen_logratios = policy_chosen_logps - policy_rejected_logps
            reject_logratios = reference_chosen_logps - reference_rejected_logps



            pi_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(reject_logratios, dim=0)
            margin_sorted = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(beta * (margin_sorted)) * (1 - label_smoothing)
                - F.logsigmoid(-beta * (margin_sorted)) * label_smoothing
            )
            chosen_reward = chosen_logratios.mean().item()
            reject_reward = reject_logratios.mean().item()
            margin=margin_sorted.mean().item()
            total_positive_reward += chosen_reward
            total_negative_reward += reject_reward
            total_margin += margin
            

            loss = losses.mean()
            total_loss += loss.item()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def AOTpair(agent, prev_model, expert_states, expert_actions, greedy=False, steps=100, beta=0.1, reject_from="random", clip_grad=False, noise_level=0.6, AOT_loss="logistic", sort_type="hard_sort", label_smoothing=0.1):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():
                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps



            pi_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(reject_logratios, dim=0)
            margin_sorted = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(beta * (margin_sorted)) * (1 - label_smoothing)
                - F.logsigmoid(-beta * (margin_sorted)) * label_smoothing
            )
            chosen_reward = chosen_logratios.mean().item()
            reject_reward = reject_logratios.mean().item()
            margin=margin_sorted.mean().item()
            total_positive_reward += chosen_reward
            total_negative_reward += reject_reward
            total_margin += margin
            

            loss = losses.mean()
            total_loss += loss.item()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def NCA(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    # beta=1
    clip_grad=True
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":

                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)
            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            with torch.no_grad():


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps

            losses=-F.logsigmoid(beta * chosen_logratios)-0.5*F.logsigmoid(-chosen_logratios)-0.5*F.logsigmoid(-reject_logratios)


            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
########################################################################################
# Reference-free Methods
########################################################################################
def SimPO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=2.0,gamma=1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    # banlancing_weight = 0.03  # You can adjust this weight

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = -torch.nn.functional.logsigmoid(beta * chosen_logratios-beta*reject_logratios-gamma)

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def CPO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = -torch.nn.functional.logsigmoid(beta * chosen_logratios-beta*reject_logratios)-chosen_logratios

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def CPO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = -torch.nn.functional.logsigmoid(beta * chosen_logratios-beta*reject_logratios)-chosen_logratios

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def CPOP(agent, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6,Lambda=500):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = -torch.nn.functional.logsigmoid(beta*Lambda * chosen_logratios-beta*reject_logratios)-chosen_logratios

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def ORPO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)
            reference_chosen_logps = (1-torch.clamp(torch.exp(policy_chosen_logps),max=-epsilon)).log()
            reference_rejected_logps = (1-torch.clamp(torch.exp(policy_rejected_logps),max=-epsilon)).log()

            positive_reward = (policy_chosen_logps - reference_chosen_logps).detach().mean().item()
            negative_reward = (policy_rejected_logps - reference_chosen_logps).detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            log_odds=(policy_chosen_logps-policy_rejected_logps)-(torch.log1p(-torch.exp(policy_chosen_logps))-torch.log1p(-torch.exp(policy_rejected_logps)))
            sig_ratio=torch.sigmoid(log_odds)
            ratio=torch.log(sig_ratio)
            losses=beta*ratio

            loss = policy_chosen_logps-losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward

def RRHF(agent, expert_states, expert_actions, greedy=False,steps=100,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = torch.clamp(-chosen_logratios+reject_logratios,min=0)-chosen_logratios

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def SLiC_HF(agent, expert_states, expert_actions, greedy=False,steps=100,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            losses = torch.clamp(1-chosen_logratios+reject_logratios,min=0)-chosen_logratios

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def CKTO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False,noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    total_KL = 0
    epsilon = 1e-3

    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)
            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)

            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            chosen_logratios = agent.ac.log_prob(state, chosen_act)
            reject_logratios = agent.ac.log_prob(state, reject_act)


            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
     
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin
            prob_KL=torch.clamp(torch.exp(chosen_logratios), max=1, min=0)
            KL_estimate = (prob_KL*torch.exp(prob_KL)).mean()
            # KL_estimate = -(chosen_logratios*torch.exp(chosen_logratios)).mean()
            total_KL += KL_estimate


            # KT0 loss
            chosen_losses =  1- torch.sigmoid(beta * (chosen_logratios - KL_estimate))
            rejected_losses = 1- torch.sigmoid(beta * (KL_estimate - reject_logratios))
            losses = torch.cat((chosen_losses, rejected_losses), 0)
            # print KL



            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.mean().item()
    print(total_KL)
    return total_loss, total_margin, total_positive_reward, total_negative_reward
# CSPPO is Mathematically incorrect
def CSPPO(agent, expert_states, expert_actions, greedy=False, steps=100, eta=1e3, reject_from="policy", clip_grad=False, noise_level=0.6):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    epsilon = 1e-3
    for i in range(steps):
            # random sample a batch of expert states and actions
            idx = np.random.randint(0, expert_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[idx]).to(agent.device)
            chosen_act = torch.FloatTensor(expert_actions[idx]).to(agent.device)

            if reject_from=="random":
                    reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(agent.device)
            elif reject_from=="policy":
                with torch.no_grad():
                    
                        reject_act, reference_rejected_logps = agent.ac(state, deterministic=False, with_logprob=True)
            elif reject_from=="add_gaussian_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.normal(0,noise_level,chosen_act.shape)).to(agent.device)
            elif reject_from=="add_noise_expert_act":
                reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-noise_level,noise_level,chosen_act.shape)).to(agent.device)           



            # Clamp the reject action to the action space
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)


            
            positive_reward = (policy_chosen_logps-eta/2).pow(2).mean()
            negative_reward = (policy_rejected_logps+eta/2).pow(2).mean()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward.item()
            total_negative_reward += negative_reward.item()
            total_margin += margin


            # SPPO loss
            losses = positive_reward+ negative_reward

            loss = losses
            total_loss += loss

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()
    total_loss=total_loss.item()
    total_margin=total_margin.item()
    return total_loss, total_margin, total_positive_reward, total_negative_reward
########################################################################################
# Entrypoint
########################################################################################
import argparse
import os 
import numpy as np
import torch
from copy import deepcopy
import gym
import pandas as pd
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for expert dataset and model parameters")

    # Required arguments
    parser.add_argument("--expert_path", type=str, required=True, help="Path to the expert dataset")
    parser.add_argument("--load_freq", type=int, default=0, help="Frequency for loading previous model")
    parser.add_argument("--method", type=str, required=True, choices=['DPO', 'KTO', 'SPPO', 'SimPO',"CPO","ORPO","RRHF","SLiC_HF","CPOP","CKTO","CSPPO","AOTpair","AOT","BCO","APOzero","APOdown","IPO","EXO","NCA","robustDPO"], help="Method to use")
    parser.add_argument("--reject_from", type=str, default="policy", choices=['random', 'policy', 'add_gaussian_noise_expert_act', 'add_noise_expert_act'], help="Method to use")
    parser.add_argument("--actor_type", type=str, default="continuous", choices=["continuous","quantile","discrete","flow","wishart"], help="Type of actor to use")
    parser.add_argument("--weight_decay", action="store_true", help="Whether to use weight decay for the optimizer")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--eval_freq", type=int, default=500, help="Evaluation frequency")

    # Optional arguments
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter (optional)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter (optional)")
    parser.add_argument("--eta", type=float, default=1e3, help="Eta parameter (optional)")
    parser.add_argument("--noise_level", type=float, default=0.6, help="Noise level for adding noise to expert actions (optional)")
    parser.add_argument("--Lambda", type=float, default=50, help="Lambda parameter (optional)")
    # seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")

    return parser.parse_args()
def get_log_path(args):
    # Create base directory structure
    base_dir = os.path.join("logs", args.env_name)
    # Different dataset diffrerent directory
    base_dir = os.path.join(base_dir, os.path.basename(args.expert_path))



    # Create method-specific filename
    filename_parts = [
        f"{args.method}",
        f"{args.actor_type}",
        f"load_freq_{args.load_freq}",
        f"random_{args.reject_from}",
        f"weight_decay_{args.weight_decay}",
        f"noise_{args.noise_level:.1f}",

    ]
    
    # Add method-specific parameters
    if args.method in ['DPO', 'KTO', 'CPO', 'ORPO', 'CKTO',"AOTpair","AOT","APOzero","APOdown","BCO","IPO","EXO","NCA","robustDPO"]:
        filename_parts.append(f"beta_{args.beta:.1f}")
    elif args.method == 'CPOP':
        filename_parts.append(f"beta_{args.beta:.1f}_Lambda_{args.Lambda:.1f}")
    elif args.method in ['SPPO', 'CSPPO']:
        filename_parts.append(f"eta_{args.eta:.1f}")
    elif args.method == 'SimPO':
        filename_parts.append(f"beta_{args.beta:.1f}_gamma_{args.gamma:.1f}")
    
    # Join all parts to create the filename
    filename = "_".join(filename_parts) + ".csv"
    
    # Ensure filename is not too long
    if len(filename) > 255:  # Max filename length for many file systems
        filename = filename[:240] + "_truncated.csv"
    
    # Combine base directory and filename
    log_path = os.path.join(base_dir, filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True, mode=0o777)
    
    return log_path
if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration
    print(f"Using expert dataset from {args.expert_path}")
    print(f"Loading previous model every {args.load_freq} steps")
    print(f"Using weight decay: {args.weight_decay}")
    print(f"Using environment: {args.env_name}")
    print(f"Total training steps: {args.total_steps}")
    print(f"Evaluation frequency: {args.eval_freq}")
    print(f"Random sample: {args.reject_from}")
    print(f"Method: {args.method}")
    print(f"Beta: {args.beta}")
    print(f"Gamma: {args.gamma}")
    print(f"Eta: {args.eta}")
    print(f"Noise level: {args.noise_level}")
    print(f"Seed: {args.seed}")
    set_seed(args.seed)

    # Load expert dataset
    # expert_obs = np.load(args.expert_path+"_obs.npy")
    # expert_act = np.load(args.expert_path+"_act.npy")

    # Initialize environment
    assert args.env_name in ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']
    expert_data=np.load(args.env_name+".npz")
    expert_obs = expert_data['states']
    expert_act = expert_data['actions']
    if Normalization:
        mean=expert_obs.mean(axis=0)
        std=expert_obs.std(axis=0)
        expert_obs=(expert_obs-mean)/std
        
    # # choose first 1k
    expert_obs = expert_obs[:1000]
    expert_act = expert_act[:1000]
    env = gym.make(args.env_name)
    env.seed(0)  # Assuming Seed is defined as 0
    # Initialize agent and previous model
    agent = Agent(expert_obs.shape[1], expert_act.shape[1], weight_decay=args.weight_decay,Actor_type=args.actor_type).to(device=device)
    actor=args.actor_type
    prev_model = deepcopy(agent.ac).to(device=device)
    all_expert_actions = torch.FloatTensor(expert_act).to(device=device)
    all_expert_states = torch.FloatTensor(expert_obs).to(device=device)

    # Initialize tracking variables
    loss_list = []
    margin_list = []
    positive_reward_list = []
    negative_reward_list = []
    return_det_list = []
    return_sto_list = []

    # Initial evaluation
    print("Step 0")
    print("Evaluating real return")
    horizon = 1000
    n_episodes = 1
    real_return_det = evaluate_real_return(agent.ac, env, n_episodes, horizon, deterministic=True)
    real_return_sto = evaluate_real_return(agent.ac, env, n_episodes, horizon, deterministic=False)
    print(f"Deterministic real return: {real_return_det}")
    print(f"Stochastic real return: {real_return_sto}")
    return_det_list.append(real_return_det)
    return_sto_list.append(real_return_sto)
    loss_list.append(torch.tensor(0))
    margin_list.append(0)
    positive_reward_list.append(0)
    negative_reward_list.append(0)


    # Main training loop
    for step in range(1,1+int(args.total_steps / args.eval_freq)):
        # Training step
        if args.method == 'DPO':
            loss, margin, positive_reward, negative_reward = DPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'EXO':
            loss, margin, positive_reward, negative_reward = EXO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'NCA':
            loss, margin, positive_reward, negative_reward = NCA(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)

        elif args.method == 'IPO':
            loss, margin, positive_reward, negative_reward = IPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'robustDPO':
            loss, margin, positive_reward, negative_reward = robustDPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)

        elif args.method == 'KTO':
            loss, margin, positive_reward, negative_reward = KTO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta,noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'SPPO':
            loss, margin, positive_reward, negative_reward = SPPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, eta=args.eta, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'AOT':
            loss, margin, positive_reward, negative_reward = AOT(agent,prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq,beta=args.beta)
        elif args.method == 'AOTpair':
            loss, margin, positive_reward, negative_reward = AOTpair(agent,prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq,beta=args.beta)
        elif args.method == 'APOzero':
            loss, margin, positive_reward, negative_reward = APOzero(agent,prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq,beta=args.beta)
        elif args.method == 'APOdown':
            loss, margin, positive_reward, negative_reward = APOdown(agent,prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq,beta=args.beta)
        elif args.method == 'BCO':
            loss, margin, positive_reward, negative_reward = BCO(agent,prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq,beta=args.beta)
        elif args.method == 'SimPO':
            loss, margin, positive_reward, negative_reward = SimPO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, gamma=args.gamma, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'CPO':
            loss, margin, positive_reward, negative_reward = CPO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'ORPO':
            loss, margin, positive_reward, negative_reward = ORPO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'RRHF':
            loss, margin, positive_reward, negative_reward = RRHF(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'SLiC_HF':
            loss, margin, positive_reward, negative_reward = SLiC_HF(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'CPOP':
            loss, margin, positive_reward, negative_reward = CPOP(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, noise_level=args.noise_level,steps=args.eval_freq,Lambda=args.Lambda)
        elif args.method == 'CKTO':
            loss, margin, positive_reward, negative_reward = CKTO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, noise_level=args.noise_level,steps=args.eval_freq)
        elif args.method == 'CSPPO':
            loss, margin, positive_reward, negative_reward = CSPPO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, eta=args.eta, noise_level=args.noise_level,steps=args.eval_freq)

        else:
            raise ValueError("Invalid method")
        
        # Update tracking variables
        loss_list.append(loss)
        margin_list.append(margin)
        positive_reward_list.append(positive_reward)
        negative_reward_list.append(negative_reward)

        # Update previous model if needed
        if args.load_freq > 0 and step % args.load_freq == 0:
            prev_model.load_state_dict(agent.ac.state_dict())


        # Evaluate if needed
        print(f"---------------------------------\nStep {step * args.eval_freq}")
        print(f"Loss: {loss}")
        print(f"Margin: {margin}")
        print(f"Positive reward: {positive_reward}")
        print(f"Negative reward: {negative_reward}")
        print("---------------------------------")
        print("Evaluating real return")
        real_return_det = evaluate_real_return(agent.ac, env, n_episodes, horizon, deterministic=True)
        real_return_sto = evaluate_real_return(agent.ac, env, n_episodes, horizon, deterministic=False)
        print(f"Deterministic real return: {real_return_det}")
        print(f"Stochastic real return: {real_return_sto}")
        return_det_list.append(real_return_det)
        return_sto_list.append(real_return_sto)
    # all to numpy  [to_cpu_numpy(item) for item in tensor_or_list]
    loss_list = np.array(loss_list)

    margin_list = np.array(margin_list)
    positive_reward_list = np.array(positive_reward_list)
    negative_reward_list = np.array(negative_reward_list)
    return_det_list = np.array(return_det_list)
    return_sto_list = np.array(return_sto_list)
    # Save results
    df = pd.DataFrame({
        'loss': loss_list,
        'margin': margin_list,
        'positive_reward': positive_reward_list,
        'negative_reward': negative_reward_list,
        'deterministic_return': return_det_list,
        'stochastic_return': return_sto_list
    })
    log_path = get_log_path(args)
    df.to_csv(log_path, index=False)
    print(f"Results saved to {log_path}")
    best_return = max(return_det_list)
    print(f"Best deterministic return: {best_return}")
    best_return = max(return_sto_list)
    print(f"Best stochastic return: {best_return}")
