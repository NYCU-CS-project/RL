import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
Seed=0
########################################################################################
# Eval 
########################################################################################
def evaluate_real_return(policy, env, n_episodes, horizon, deterministic):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0

        for t in range(horizon):
            state=torch.FloatTensor(obs).to(policy.device)
            action = policy(state, deterministic, with_logprob=False)
            # to numpy
            action = (action[0]).cpu().detach().numpy()
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

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim,hidden_sizes, activation=nn.ReLU, act_limit=1):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.device = 'cuda'

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

        act=torch.atanh(torch.Tensor(act).to(obs.device))

        logp_pi = pi_distribution.log_prob(act).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(axis=1)

        return logp_pi
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256), activation=nn.ReLU, act_limit=1,weight_decay=False,device='cuda'):
        super().__init__()
        self.ac=Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        if weight_decay:
            self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=3e-4, weight_decay=1e-4)
        else:
            self.pi_optimizer = torch.optim.Adam(self.ac.parameters(), lr=3e-4)
        self.device = device
########################################################################################
##Methods
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
def DPO(agent, prev_model, expert_states, expert_actions, greedy=False,steps=100,beta=0.1,reject_from="random",clip_grad=False):
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
                with torch.no_grad():
                    reject_act = chosen_act + torch.randn_like(chosen_act) * 0.1
            elif reject_from=="add_noise_expert_act":
                with torch.no_grad():
                    reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-0.1,0.1,chosen_act.shape)).to(agent.device)
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
            losses = -torch.nn.functional.logsigmoid(beta * logits)

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def KTO(agent, prev_model, expert_states, expert_actions,greedy=False, steps=100, beta=0.1, reject_from="random", clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
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
                with torch.no_grad():
                    reject_act = chosen_act + torch.randn_like(chosen_act) * 0.1
            elif reject_from=="add_noise_expert_act":
                with torch.no_grad():
                    reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-0.1,0.1,chosen_act.shape)).to(agent.device)

            
            # Clamp the reject action to the action space
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = agent.ac.log_prob(state, chosen_act)
            policy_rejected_logps = agent.ac.log_prob(state, reject_act)

            # Calculate KL divergence
            net_out = agent.ac.net(state)
            policy_mu = agent.ac.mu_layer(net_out)
            policy_log_std = torch.clamp(agent.ac.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            policy_std = torch.exp(policy_log_std)



            with torch.no_grad():
                prev_net_out = prev_model.net(state)
                prev_mu = prev_model.mu_layer(prev_net_out)
                prev_log_std = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
                prev_std = torch.exp(prev_log_std)


                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(state, reject_act)

            kl_divergence = gaussian_kl_divergence(policy_mu, policy_std, prev_mu, prev_std)


            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps

            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward

            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence))
            rejected_losses = 1 - torch.sigmoid(beta * (kl_divergence - reject_logratios))

            losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
            loss = losses.mean()

            total_loss += loss.item()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def SPPO(agent, prev_model, expert_states, expert_actions, greedy=False, steps=100, eta=1e3, reject_from="random", clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
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
                with torch.no_grad():
                    reject_act = chosen_act + torch.randn_like(chosen_act) * 0.1
            elif reject_from=="add_noise_expert_act":
                with torch.no_grad():
                    reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-0.1,0.1,chosen_act.shape)).to(agent.device)                    



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
            
            positive_reward = (chosen_logratios-eta/2).pow(2).detach().mean().item()
            negative_reward = (reject_logratios+eta/2).pow(2).detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin


            # SPPO loss
            losses = (chosen_logratios-eta/2).pow(2)+ (reject_logratios+eta/2).pow(2)

            loss = losses.mean()
            total_loss += loss.sum()

            agent.pi_optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_norm=1.0)
            agent.pi_optimizer.step()

    return total_loss, total_margin, total_positive_reward, total_negative_reward

def SimPO(agent, expert_states, expert_actions, greedy=False,steps=100,beta=2.0,gamma=1,reject_from="random",clip_grad=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
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
                with torch.no_grad():
                    # data + torch.randn_like(data) * std + mean
                    # std=0.1 mean=0
                    reject_act = chosen_act + torch.randn_like(chosen_act) * 0.1
            elif reject_from=="add_noise_expert_act":
                with torch.no_grad():
                    reject_act = chosen_act + torch.FloatTensor(np.random.uniform(-0.1,0.1,chosen_act.shape)).to(agent.device)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for expert dataset and model parameters")

    # Required arguments
    parser.add_argument("--expert_path", type=str, required=True, help="Path to the expert dataset")
    parser.add_argument("--load_freq", type=int, required=True, help="Frequency for loading previous model")
    parser.add_argument("--method", type=str, required=True, choices=['DPO', 'KTO', 'SPPO', 'SimPO'], help="Method to use")
    parser.add_argument("--reject_from", type=str, default="random", choices=['random', 'policy', 'add_gaussian_noise_expert_act', 'add_noise_expert_act'], help="Method to use")
    parser.add_argument("--weight_decay", action="store_true", help="Whether to use weight decay for the optimizer")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the environment")
    parser.add_argument("--total_steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency")

    # Optional arguments
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter (optional)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter (optional)")
    parser.add_argument("--eta", type=float, default=1e3, help="Eta parameter (optional)")

    return parser.parse_args()
def get_log_path(args):
    # Create base directory structure
    base_dir = os.path.join("logs", args.env_name)
    
    # Create method-specific filename
    filename_parts = [
        f"{args.method}",
        f"load_freq_{args.load_freq}",
        f"random_{args.reject_from}",
        f"weight_decay_{args.weight_decay}",

    ]
    
    # Add method-specific parameters
    if args.method in ['DPO', 'KTO']:
        filename_parts.append(f"beta_{args.beta:.1f}")
    elif args.method == 'SPPO':
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

    # Load expert dataset
    expert_obs = np.load(args.expert_path+"_obs.npy")
    expert_act = np.load(args.expert_path+"_act.npy")

    # Initialize environment
    assert args.env_name in ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']
    env = gym.make(args.env_name)
    env.seed(0)  # Assuming Seed is defined as 0

    # Initialize agent and previous model
    agent = Agent(expert_obs.shape[1], expert_act.shape[1], weight_decay=args.weight_decay, device='cuda').to('cuda')
    prev_model = deepcopy(agent.ac).to('cuda')

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
    n_episodes = 5
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
    for step in range(int(args.total_steps / args.eval_freq)):
        # Training step
        if args.method == 'DPO':
            loss, margin, positive_reward, negative_reward = DPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta)
        elif args.method == 'KTO':
            loss, margin, positive_reward, negative_reward = KTO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta)
        elif args.method == 'SPPO':
            loss, margin, positive_reward, negative_reward = SPPO(agent, prev_model, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, eta=args.eta)
        elif args.method == 'SimPO':
            loss, margin, positive_reward, negative_reward = SimPO(agent, expert_obs, expert_act, reject_from=args.reject_from, clip_grad=False, beta=args.beta, gamma=args.gamma)
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
    loss_list = np.array([item.detach().numpy for item in loss_list])
    margin_list = np.array([item for item in margin_list])
    positive_reward_list = np.array([item for item in positive_reward_list])
    negative_reward_list = np.array([item for item in negative_reward_list])
    return_det_list = np.array([item for item in return_det_list])
    return_sto_list = np.array([item for item in return_sto_list])
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
