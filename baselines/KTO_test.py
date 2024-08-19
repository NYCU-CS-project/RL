# '''
# Behavior cloning MLE(Learnt variance) and (MSE)Fixed variance policy.
# '''

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
import torch.nn.functional as F
from common.sac import ReplayBuffer, SAC
import matplotlib.pyplot as plt
import envs
from utils import system, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy
is_ant=False
def try_evaluate(itr: int, policy_type: str):
    assert policy_type in ["Running"]
    update_time = itr * v['bc']['eval_freq']


    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], True,is_ant)

    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], False,is_ant)

    print(f"real sto return avg: {real_return_sto:.2f}")
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

    logger.record_tabular(f"{policy_type} Update Time", update_time)

    return real_return_det, real_return_sto


def stochastic_bc(sac_agent, expert_states, expert_actions, epochs = 100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 1000
    total_loss = 0
    for i in range(epochs):
        for batch_no in range(expert_states.shape[0]//batch_size):
            start_id = batch_no*batch_size
            end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
    #         log_pi = sac_agent.ac.pi.log_prob(torch.FloatTensor(expert_states[start_id:end_id,:]),\
    #                                                         torch.FloatTensor(expert_actions[start_id:end_id,:]))
    #         sac_agent.pi_optimizer.zero_grad()
    #         nll = -(log_pi).mean()
    #         total_loss+=-(log_pi).sum()
    #         nll.backward()
    #         sac_agent.pi_optimizer.step()

    # total_loss = total_loss/(epochs*expert_states.shape[0])
            state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)



            
            # Calculate log probabilities for chosen actions
            
            log_pi = sac_agent.ac.pi.log_prob(state, chosen_act)
            
 
            losses = -(log_pi)
            loss = losses.mean()
            # check nan

            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()


    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss        
from torch.distributions import Normal       
def entropy(sac_agent, obs):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    net_out = sac_agent.net(obs)
    mu = sac_agent.mu_layer(net_out)
    log_std = sac_agent.log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    # 计算高斯分布的熵
    gaussian_entropy = 0.5 * torch.log(2 * np.pi * np.e * std**2+1e-6).sum(-1)

    # 计算 tanh 变换的修正项
    pi_action = Normal(mu, std).rsample()
    log_det_jacobian = (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(-1)
    
    # 总熵 = 高斯熵 - log_det_jacobian
    entropy = gaussian_entropy - log_det_jacobian

    return entropy
def DPO(sac_agent, prev_model, expert_states, expert_actions, greedy=False,epochs=100,random_prob=0.3):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 1000
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.5
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    lambda_entropy = 0.001
    epsilon = 1e-6

    for i in range(epochs):
        for batch_no in range(expert_states.shape[0] // batch_size):
            start_id = batch_no * batch_size
            end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
            state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)
            chosen_act= torch.clamp(chosen_act, -1+epsilon, 1-epsilon)

            # Use the forward method for sampling
            if random_prob>0 and np.random.rand()<random_prob:
                reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(sac_agent.device)
            else:
                with torch.no_grad():
                    # Use the forward method for sampling
                    if greedy:
                        reject_act, reference__rejected_logps = prev_model(state, deterministic=True, with_logprob=True)
                    else:
                        reject_act, reference_rejected_logps = prev_model(state, deterministic=False, with_logprob=True)
            # Clamp the reject action to the action space
            
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)

            # Calculate KL divergence
            net_out = sac_agent.ac.pi.net(state)
            policy_mu = sac_agent.ac.pi.mu_layer(net_out)
            policy_log_std = torch.clamp(sac_agent.ac.pi.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
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


            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps
            # u1= torch.exp(policy_chosen_logps)/(torch.exp(reference_chosen_logps)+1e-6)+1e-6
            # u2= torch.exp(policy_rejected_logps)/(torch.exp(reference_rejected_logps)+1e-6)+1e-6


            # reverse kl(original DPO)
            losses = -torch.nn.functional.logsigmoid(beta * logits)

            # hinge loss
            # losses = torch.nn.functional.relu(1 - beta * logits)

            # forward kl= -logsigmoid(-beta*(1/u1) + beta*(1/u2))
            # losses = -torch.nn.functional.logsigmoid(-beta*(1/u1) + beta*(1/u2))

            # JS divergence
            # losses = -torch.nn.functional.logsigmoid(beta *(torch.log(2*u1/1+u1)-torch.log(2*u2/1+u2)))
            # losses = -torch.nn.functional.logsigmoid(beta *(chosen_logratios - reject_logratios-torch.log(1+u1)+torch.log(1+u2)))

            # alpha divergence
            alpha=0.5 # alpha is in (0,1)
            # losses = -torch.nn.functional.logsigmoid(beta *((1-u1**(-alpha))/(alpha)-(1-u2**(-alpha))/(alpha)))

            # Total variation
            #  u1>1? 1/2:-1/2
            #  u2>1? 1/2:-1/2
            # losses = -torch.nn.functional.logsigmoid(beta *(u1>1).float()-(u2>1).float())


            # chi-squared divergence
            # losses = -torch.nn.functional.logsigmoid(beta *(2*u1-2*u2))

            




            # loss = losses.mean()+kl_divergence
            # loss = losses.mean()+0.97*policy_chosen_logps.mean()
            # list the function of sac_agent.ac.pi

            # loss = losses.mean()+kl_divergence - lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            # loss=losses.mean()-lambda_entropy * sac_agent.ac.pi.entropy(state).mean()

            # real gail rewards

            # L = logistic loss
            # losses= -torch.nn.functional.logsigmoid(beta*logits)
            
            # L = tanh loss
            # losses = torch.nn.functional.tanh(logits)
            # L = hinge loss
            # losses = torch.nn.functional.relu(1 - beta * logits)
            # exponential loss
            # losses = -torch.exp(-beta * logits)
            # L = Huber loss
            # losses = torch.nn.functional.smooth_l1_loss(logits, torch.zeros_like(logits))

            # L = SPPO loss
            # losses = (beta*chosen_logratios-1/2)**2+(beta*reject_logratios+1/2)**2


            loss = losses.mean()
            # loss=losses.mean()+kl_divergence+lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            # loss=losses.mean()+lambda_entropy * entropy(sac_agent.ac.pi, state).mean()

            # check nan
            if torch.isnan(loss).any():
                print("nan in loss")
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            # check 0
            if torch.isnan(sac_agent.ac.pi.log_std_layer.weight.grad).any():
                print("nan in grad")
            torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward


def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two multivariate Gaussian distributions.
    Assumes diagonal covariance matrices.
    """
    var1 = sigma1.pow(2)
    var2 = sigma2.pow(2)
    kl = (var1 / var2 + (mu2 - mu1).pow(2) / var2 - 1 + var2.log() - var1.log()).sum(-1) * 0.5
    return kl.mean()
def strange_DPO(sac_agent, prev_model, expert_states, expert_actions, greedy=False,epochs=100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 1000
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1
    total_demo_loss=0

    for i in range(epochs):
        # for batch_no in range(expert_states.shape[0] // batch_size):
        #     start_id = batch_no * batch_size
        #     end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
        #     state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
        #     chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)
            
        #     policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
        #     with torch.no_grad():
                
        #         reference_chosen_logps = prev_model.log_prob(state, chosen_act)

        #     losses = -(policy_chosen_logps - reference_chosen_logps)
        #     loss = losses.mean()
        #     # check nan
        #     total_demo_loss += loss.sum()

        #     sac_agent.pi_optimizer.zero_grad()
        #     loss.backward()
        #     # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
        #     sac_agent.pi_optimizer.step()
    
        for batch_no in range(expert_states.shape[0] // batch_size):
            start_id = batch_no * batch_size
            end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
            state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)
            # check nan
            if torch.isnan(chosen_act).any():
                print("nan in expert actions")
                

            # Use the new forward method for sampling
            if greedy:
                reject_act, policy_rejected_logps = sac_agent.ac.pi(state, deterministic=True, with_logprob=True)
            else:
                reject_act, policy_rejected_logps = sac_agent.ac.pi(state, deterministic=False, with_logprob=True)
            #clamp the reject action to the action space
            epsilon=1e-6
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            # check nan
            if torch.isnan(reject_act).any():
                print("nan in reject actions")
            
            # Calculate log probabilities for chosen actions
            
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            # check nan
            if torch.isnan(policy_chosen_logps).any():
                print("nan in policy_chosen_logps")

            with torch.no_grad():
                
                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                # check nan
                if torch.isnan(reference_chosen_logps).any():
                    print("nan in reference_chosen_logps")
                
                reference_rejected_logps = prev_model.log_prob(state, reject_act)
                # check nan
                if torch.isnan(reference_rejected_logps).any():
                    print("nan in reference_rejected_logps")
                    print(reject_act)
                    print(reference_rejected_logps)
                    #print the reject where reference_rejected_logps is nan
                    print(reject_act[torch.isnan(reference_rejected_logps)]) 

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            
            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward
            
            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps
            # losses = -torch.nn.functional.logsigmoid(beta * logits)-(policy_chosen_logps-reference_chosen_logps)
            # hinge loss
            losses = torch.nn.functional.relu(1 - beta * logits)-(policy_chosen_logps-reference_chosen_logps)
            loss = losses.mean()
            # check nan
            if torch.isnan(loss).any():
                print("nan in loss")
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward, total_demo_loss
def KTO(sac_agent, prev_model, expert_states, expert_actions,imperfect_states,imperfect_actions, greedy=False, epochs=100,whether_entropy=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1
    desirable_weight = 1.0  # You can adjust this weight
    undesirable_weight = 1.0  # You can adjust this weight
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    lambda_entropy = 0.0001

    for i in range(epochs):
        for batch_no in range(1):
            # start_id = batch_no * batch_size
            # end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
            # state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            # chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)
            # randomly sample a batch of expert and imperfect data seperately
            expert_batch = np.random.randint(0, expert_states.shape[0], batch_size)
            imperfect_batch = np.random.randint(0, imperfect_states.shape[0], batch_size)
            state = torch.FloatTensor(expert_states[expert_batch, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[expert_batch, :]).to(sac_agent.device)
            reject_state = torch.FloatTensor(imperfect_states[imperfect_batch, :]).to(sac_agent.device)
            reject_act = torch.FloatTensor(imperfect_actions[imperfect_batch, :]).to(sac_agent.device)


            
            # Clamp the reject action to the action space
            epsilon = 1e-6
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(reject_state, reject_act)

            # Calculate KL divergence
            net_out = sac_agent.ac.pi.net(state)
            policy_mu = sac_agent.ac.pi.mu_layer(net_out)
            policy_log_std = torch.clamp(sac_agent.ac.pi.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            policy_std = torch.exp(policy_log_std)

            net_out = sac_agent.ac.pi.net(reject_state)
            policy_mu_reject = sac_agent.ac.pi.mu_layer(net_out)
            policy_log_std_reject = torch.clamp(sac_agent.ac.pi.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
            policy_std_reject = torch.exp(policy_log_std)


            with torch.no_grad():
                prev_net_out = prev_model.net(state)
                prev_mu = prev_model.mu_layer(prev_net_out)
                prev_log_std = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
                prev_std = torch.exp(prev_log_std)
                prev_net_out = prev_model.net(reject_state)
                prev_mu_reject = prev_model.mu_layer(prev_net_out)
                prev_log_std_reject = torch.clamp(prev_model.log_std_layer(prev_net_out), LOG_STD_MIN, LOG_STD_MAX)
                prev_std_reject = torch.exp(prev_log_std_reject)

                reference_chosen_logps = prev_model.log_prob(state, chosen_act)
                reference_rejected_logps = prev_model.log_prob(reject_state, reject_act)

            kl_divergence_chosen = gaussian_kl_divergence(policy_mu, policy_std, prev_mu, prev_std)
            kl_divergence_rejected = gaussian_kl_divergence(policy_mu_reject, policy_std_reject, prev_mu_reject, prev_std_reject)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps

            positive_reward = chosen_logratios.detach().mean().item()
            negative_reward = reject_logratios.detach().mean().item()
            margin = positive_reward - negative_reward

            total_positive_reward += positive_reward
            total_negative_reward += negative_reward
            total_margin += margin

            # Corrected KTO loss calculation
            # L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
            chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence_chosen))
            # chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence))
            rejected_losses = 1 - torch.sigmoid(beta * (kl_divergence_rejected - reject_logratios))

            losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
            # losses = chosen_losses
            if whether_entropy:
                loss = losses.mean()- lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            else:
                loss = losses.mean()
            # loss = losses.mean()- lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            # loss -= lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            # loss=losses.mean()

            total_loss += loss.item()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def strange_BC(sac_agent, prev_model, expert_states, expert_actions, greedy=False,epochs=100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 1000
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1

    for i in range(epochs):
        for batch_no in range(expert_states.shape[0] // batch_size):
            start_id = batch_no * batch_size
            end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
            state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)
            
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            with torch.no_grad():
                
                reference_chosen_logps = prev_model.log_prob(state, chosen_act)

            losses = -(policy_chosen_logps - reference_chosen_logps)
            loss = losses.mean()
            # check nan
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss
def mse_bc(sac_agent, expert_states, expert_actions, epochs = 100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 1000
    total_loss = 0
    for i in range(epochs):
        for batch_no in range(expert_states.shape[0]//batch_size):
            start_id = batch_no*batch_size
            end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
            state= torch.FloatTensor(expert_states[start_id:end_id,:]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id,:]).to(sac_agent.device)
            # mse
            se=((sac_agent.ac.pi(state)[0]-chosen_act)**2).sum(1)

            # se = ((sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]))[0] - torch.FloatTensor(expert_actions[start_id:end_id,:]))**2).sum(1)
            loss = se.mean()
            sac_agent.pi_optimizer.zero_grad()
            total_loss+=se.sum()
            loss.backward()
            sac_agent.pi_optimizer.step()

    total_loss = total_loss/(epochs*expert_states.shape[0])

    return total_loss               
def strange_KTO(sac_agent, prev_model, expert_states, expert_actions, greedy=False, epochs=100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 1000
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1
    desirable_weight = 1.0  # You can adjust this weight
    undesirable_weight = 1.0  # You can adjust this weight
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    total_demo_loss = 0
    lambda_entropy = 0.0001

    for i in range(epochs):
        # for batch_no in range(expert_states.shape[0] // batch_size):
        #     start_id = batch_no * batch_size
        #     end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
        #     state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
        #     chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)

        #     policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
        #     with torch.no_grad():
        #         reference_chosen_logps = prev_model.log_prob(state, chosen_act)

        #     losses = -(policy_chosen_logps - reference_chosen_logps)
        #     loss = losses.mean()
        #     # check nan
        #     total_demo_loss += loss.sum()

        #     sac_agent.pi_optimizer.zero_grad()
        #     loss.backward()
        #     # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
        #     sac_agent.pi_optimizer.step()

        for batch_no in range(expert_states.shape[0] // batch_size):
            start_id = batch_no * batch_size
            end_id = min((batch_no + 1) * batch_size, expert_states.shape[0])
            state = torch.FloatTensor(expert_states[start_id:end_id, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[start_id:end_id, :]).to(sac_agent.device)

            # # Use the forward method for sampling
            # if greedy:
            #     reject_act, policy_rejected_logps = sac_agent.ac.pi(state, deterministic=True, with_logprob=True)
            # else:
            #     reject_act, policy_rejected_logps = sac_agent.ac.pi(state, deterministic=False, with_logprob=True)
            # uniformly random a  reject action
            reject_act = torch.FloatTensor(np.random.uniform(-1, 1, chosen_act.shape)).to(sac_agent.device)

            # Clamp the reject action to the action space
            epsilon = 1e-6
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)

            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)

            # Calculate KL divergence
            net_out = sac_agent.ac.pi.net(state)
            policy_mu = sac_agent.ac.pi.mu_layer(net_out)
            policy_log_std = torch.clamp(sac_agent.ac.pi.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
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

            # Corrected KTO loss calculation
            chosen_losses = 1 - torch.sigmoid(beta * (chosen_logratios - kl_divergence))
            rejected_losses = 1 - torch.sigmoid(beta * (kl_divergence - reject_logratios))

            losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
            # print(kl_divergence.mean().item())
            # print kl_divergence
            print("KL divergence: ", kl_divergence.mean().item())

            loss = losses.mean()
            # loss-=lambda_entropy * entropy(sac_agent.ac.pi, state).mean()
            total_loss += loss.item()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            sac_agent.pi_optimizer.step()
            

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward, total_demo_loss

def SPPO_from_rand(sac_agent, prev_model, expert_states, expert_actions, greedy=False, epochs=100,whether_entropy=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    # lambda_entropy = 0.0001
    gap_weight=1e3



    for i in range(epochs):
        for batch_no in range(1):
            expert_batch = np.random.randint(0, expert_states.shape[0], batch_size)

            state = torch.FloatTensor(expert_states[expert_batch, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[expert_batch, :]).to(sac_agent.device)
            reject_act = torch.FloatTensor(np.random.uniform(-1,1,chosen_act.shape)).to(sac_agent.device)


            
            # Clamp the reject action to the action space
            epsilon = 1e-6
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)
            reference_chosen_logps = prev_model.log_prob(state, chosen_act)
            reference_rejected_logps = prev_model.log_prob(state, reject_act)
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            loss= (beta*chosen_logratios-gap_weight/2)**2+(beta*reject_logratios+gap_weight/2)**2
            total_loss += loss.item()
            total_margin += (chosen_logratios - reject_logratios).mean().item()
            total_positive_reward += chosen_logratios.mean().item()
            total_negative_reward += reject_logratios.mean().item()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward
def SPPO_from_prev(sac_agent, prev_model, expert_states, expert_actions, greedy=False, epochs=100,whether_entropy=False):
    assert expert_states.shape[0] == expert_actions.shape[0]
    prev_model.eval()
    batch_size = 256
    total_loss = 0
    total_margin = 0
    total_positive_reward = 0
    total_negative_reward = 0
    beta = 0.1

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    # lambda_entropy = 0.0001
    gap_weight=1e3



    for i in range(epochs):
        for batch_no in range(1):
            expert_batch = np.random.randint(0, expert_states.shape[0], batch_size)

            state = torch.FloatTensor(expert_states[expert_batch, :]).to(sac_agent.device)
            chosen_act = torch.FloatTensor(expert_actions[expert_batch, :]).to(sac_agent.device)
            with torch.no_grad():
                reject_act, policy_rejected_logps = prev_model(state, deterministic=False, with_logprob=True)


            
            # Clamp the reject action to the action space
            epsilon = 1e-6
            reject_act = torch.clamp(reject_act, -1+epsilon, 1-epsilon)
            chosen_act = torch.clamp(chosen_act, -1+epsilon, 1-epsilon)
            # Calculate log probabilities for chosen actions
            policy_chosen_logps = sac_agent.ac.pi.log_prob(state, chosen_act)
            policy_rejected_logps = sac_agent.ac.pi.log_prob(state, reject_act)
            reference_chosen_logps = prev_model.log_prob(state, chosen_act)
            reference_rejected_logps = prev_model.log_prob(state, reject_act)
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            reject_logratios = policy_rejected_logps - reference_rejected_logps
            loss= (beta*chosen_logratios-gap_weight/2)**2+(beta*reject_logratios+gap_weight/2)**2
            total_loss += loss.item()
            total_margin += (chosen_logratios - reject_logratios).mean().item()
            total_positive_reward += chosen_logratios.mean().item()
            total_negative_reward += reject_logratios.mean().item()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

    prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    # argv[2] is method
    is_DPO = False
    is_KTO = False
    if sys.argv[2] == "DPO_greedy":
        method = "DPO_greedy"
        is_DPO = True
    elif sys.argv[2] == "DPO_stochastic":
        method = "DPO_stochastic"
        is_DPO = True
    elif sys.argv[2] == "KTO_greedy":
        method = "KTO_greedy"
        is_KTO = True
    elif sys.argv[2] == "KTO_stochastic":
        method = "KTO_stochastic"
        is_KTO = True
    elif sys.argv[2] == "KTO_greedy_warmup":
        method = "KTO_greedy_warmup"
        is_KTO = True
    elif sys.argv[2] == "KTO_stochastic_warmup":
        method = "KTO_stochastic_warmup"
        is_KTO = True
    elif sys.argv[2] == "DPO_stochastic_warmup":
        method = "DPO_stochastic_warmup"
        is_DPO = True
    elif sys.argv[2] == "DPO_greedy_warmup":
        method = "DPO_greedy_warmup"
        is_DPO = True
    elif sys.argv[2] == "NLL":
        method = "NLL"
    elif sys.argv[2] == "MSE":
        method = "MSE"
    elif sys.argv[2] == "strange_DPO":
        method = "strange_DPO"
        is_DPO = True
    elif sys.argv[2] == "strange_KTO":
        method = "strange_KTO"
        is_KTO = True
    elif sys.argv[2] == "strange_BC":
        method = "strange_BC"
    elif sys.argv[2] == "strange_KTO_warmup":
        method = "strange_KTO_warmup"
        is_KTO = True
    elif sys.argv[2] == "strange_DPO_warmup":
        method = "strange_DPO_warmup"
        is_DPO = True
    elif sys.argv[2] == "SPPO_from_rand":
        method = "SPPO_from_rand"
        is_KTO = True
    elif sys.argv[2] == "SPPO_from_prev":
        method = "SPPO_from_prev"
        is_KTO = True
    else:
        # throw error
        print("Invalid method")
        exit(1)
    # argv[3] is num_expert_trajs
    num_expert_trajs = int(sys.argv[3])
    # argv[4] is num_imperfect_trajs
    num_imperfect_trajs = int(sys.argv[4])
    warmup_itr = 10
    prev_load_freq = 4000000
    best_score_det = -1e6
    best_score_sto = -1e6
    if "SPPO" in method:
        prev_load_freq=1



        

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    # num_expert_trajs = v['bc']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    print(f"device: {device}")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['bc']

    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    # os.system(f'cp baselines/bc.py {log_folder}')
    # os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    if env_name == 'AntFH-v0':
        is_ant = True
        
        
    #     env_fn = lambda: gym.make("Ant-v3")
    #     print("ok")
    # type of environment

    gym_env = env_fn()

    # gym_env =  gym.make(env_name)

    # state= gym_env.reset()
    # print(gym_env)
    # print(gym_env.action_space)
    # print(gym_env.observation_space)
    # gym_env = gym.make(env_name)
    # exit()

    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # # load expert samples from trained policy
    # print(torch.load(f'expert_data/states/{env_name}_airl.pt').numpy().shape)
    # expert_state_trajs = torch.load(f'expert_data/states/{env_name}_airl.pt').numpy()[:, :, state_indices]
    # print(torch.load(f'expert_data/states/{env_name}_airl.pt').numpy().shape)

    # expert_state_trajs = expert_state_trajs[:num_expert_trajs, :-1, :] # select first expert_episodes
    # expert_states = expert_state_trajs.copy().reshape(-1, len(state_indices))
    # print(expert_state_trajs.shape, expert_states.shape) # ignored starting state

    # expert_action_trajs = torch.load(f'expert_data/actions/{env_name}_airl.pt').numpy()
    # expert_action_trajs = expert_action_trajs[:num_expert_trajs, 1:, :] # select first expert_episodes
    # expert_actions = expert_action_trajs.reshape(-1, gym_env.action_space.shape[0])
    # load     expert_data = {'initial_states': expert_initial_states,
    #                'states': expert_states,
    #                'actions': expert_actions,
    #                'next_states': expert_next_states,
    #                'dones': expert_dones}
    # with open(f'{env_id}_{expert_dataset_name}_{expert_num_traj}.pt', 'wb') as f:
    #     pickle.dump(expert_data, f)

    # imperfect_data = {'initial_states': imperfect_init_states,
    #                     'states': imperfect_states,
    #                     'actions': imperfect_actions,
    #                     'next_states': imperfect_next_states,
    #                     'dones': imperfect_dones}
    # with open(f'{env_id}_{imperfect_dataset_names}_{imperfect_num_trajs}.pt', 'wb') as f:
    #     pickle.dump(imperfect_data, f)


    
    # expert_data = torch.load(f'baselines/Hopper_expert_400.pt')
    # expert_states = expert_data['states']
    # expert_actions = expert_data['actions']
    # imperfect_data = torch.load(f'baselines/Hopper_random_1600.pt')
    # imperfect_states = imperfect_data['states']
    # imperfect_actions = imperfect_data['actions']
    en=env_name.split('F')[0]
    normalize_greedy_score = []
    normalize_stochastic_score = []
    if en=="Hopper":
        expert_score=3607.890
        random_score=832.351
    elif en=="Walker2d":
        expert_score=4924.278
        random_score=91.524
    elif en=="HalfCheetah":
        expert_score=10656.426
        random_score= -288.797
    elif en=="Ant":
        expert_score=4778.389
        random_score=-388.064
    # num_expert_trajs=400
    # num_imperfect_trajs=400
    import pickle
    expert_data = pickle.load(open(f'/mnt/nfs/work/c98181/imitation-dice/{en}_expert_{num_expert_trajs}.pt', 'rb'))
    expert_states = expert_data['states']
    expert_actions = expert_data['actions']
    imperfect_data = pickle.load(open(f'/mnt/nfs/work/c98181/imitation-dice/{en}_random_{num_imperfect_trajs}.pt', 'rb'))
    imperfect_states = imperfect_data['states']
    imperfect_actions = imperfect_data['actions']
    import random
    if is_ant:
        # for all obs:obs = np.concatenate((obs[:27], [0.]), -1)
        expert_states = np.concatenate((expert_states[:,:27],np.zeros((expert_states.shape[0],1)),expert_states[:,27:]),-1)
        imperfect_states = np.concatenate((imperfect_states[:,:27],np.zeros((imperfect_states.shape[0],1)),imperfect_states[:,27:]),-1)

    print(expert_states.shape, expert_actions.shape)
    print(imperfect_states.shape, imperfect_actions.shape)
    expert_transition_num=1000
    imperfect_transition_num=2400
    # random_index=np.random.randint(0,expert_states.shape[0],expert_transition_num)
    # expert_states = expert_states[random_index,:]
    # expert_actions = expert_actions[random_index,:]
    # random_index=np.random.randint(0,imperfect_states.shape[0],imperfect_transition_num)
    # imperfect_states = imperfect_states[random_index,:]
    # imperfect_actions = imperfect_actions[random_index,:]
    # select first
    # expert_states = expert_states[3000:expert_transition_num+3000,:]
    # expert_actions = expert_actions[3000:expert_transition_num+3000,:]
    # imperfect_states = imperfect_states[:imperfect_transition_num,:]
    # imperfect_actions = imperfect_actions[:imperfect_transition_num,:]
    print(expert_states.shape, expert_actions.shape)
    print(imperfect_states.shape, imperfect_actions.shape)
    # exit(0)
    graph_dir = f"baselines/{env_name}/exp_{num_expert_trajs}/random_{num_imperfect_trajs}/{method}"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir, exist_ok=True,mode=0o777)
    replay_buffer = ReplayBuffer(
                    state_size, 
                    action_size,
                    device=device,
                    size=v['sac']['buffer_size'])
    sac_agent = SAC(env_fn, replay_buffer,
        steps_per_epoch=v['env']['T'],
        update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
        max_ep_len=v['env']['T'],
        seed=seed,
        start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
        reward_state_indices=state_indices,
        device=device,
        **v['sac']
    )

    loss_graph = []
    real_return_det_graph = []
    real_return_sto_graph = []
    # method = 'NLL'
    # method = "MSE"
    # method = "DPO_greedy"
    # method = "DPO_stochastic"
    if is_DPO or is_KTO or method=="strange_BC":
        prev_model = copy.deepcopy(sac_agent.ac.pi)
        margin_graph = []
        positive_reward_graph = []
        negative_reward_graph = []
        prev_model.to(device)
    for itr in range(v['bc']['epochs']//v['bc']['eval_freq']):
        if method == 'NLL':
            loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
        elif method == "MSE":
            loss = mse_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
        elif method == "strange_BC":
            loss = strange_BC(sac_agent, prev_model, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
        elif method == "DPO_greedy":
            prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=True,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "DPO_stochastic":
            prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "DPO_greedy_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg = DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=True,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "DPO_stochastic_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg = DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "KTO_greedy":
            prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = KTO(sac_agent,prev_model, expert_states, expert_actions, greedy=True,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "KTO_stochastic":
            if itr%prev_load_freq == 0:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = KTO(sac_agent,prev_model, expert_states, expert_actions,imperfect_states=imperfect_states,imperfect_actions=imperfect_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "KTO_greedy_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg = KTO(sac_agent,prev_model, expert_states, expert_actions, greedy=True,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "KTO_stochastic_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg = KTO(sac_agent,prev_model, expert_states, expert_actions,imperfect_actions=imperfect_actions,imperfect_states=imperfect_states, greedy=False,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "strange_DPO":
            if itr%prev_load_freq == 0:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg,demons = strange_DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "strange_KTO":
            if itr%prev_load_freq == 0:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg,demons = strange_KTO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "strange_BC":
            loss = strange_BC(sac_agent, prev_model, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
        elif method == "strange_KTO_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg,demons = strange_KTO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "strange_DPO_warmup":
            if itr < warmup_itr:
                loss = stochastic_bc(sac_agent, expert_states, expert_actions, epochs = v['bc']['eval_freq'])
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            else:
                # prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
                loss,margin,pos,neg,demons = strange_DPO(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
                positive_reward_graph.append(pos)
                negative_reward_graph.append(neg)
                margin_graph.append(margin)
        elif method == "SPPO_from_rand":
            if itr%prev_load_freq == 0:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = SPPO_from_rand(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        elif method == "SPPO_from_prev":
            if itr%prev_load_freq == 0:
                prev_model.load_state_dict(sac_agent.ac.pi.state_dict())
            loss,margin,pos,neg = SPPO_from_prev(sac_agent,prev_model, expert_states, expert_actions, greedy=False,epochs = v['bc']['eval_freq'])
            positive_reward_graph.append(pos)
            negative_reward_graph.append(neg)
            margin_graph.append(margin)
        

        # draw loss real return det and sto
        
        if is_KTO:
            if method =="strange_KTO":
                logger.record_tabular("strange KTO loss", loss)
                logger.record_tabular("strange KTO margin", margin)
                logger.record_tabular("strange KTO positive reward", pos)
                logger.record_tabular("strange KTO negative reward", neg)
                logger.record_tabular("strange KTO demonstration loss", demons)
                loss_graph.append(loss)
            else:
                if "warmup" in method and itr < warmup_itr:
                    logger.record_tabular("KTO warmup loss", loss.item())
                    loss_graph.append(loss.item())
                else:
                    logger.record_tabular("KTO loss", loss)
                    logger.record_tabular("KTO margin", margin)
                    logger.record_tabular("KTO positive reward", pos)
                    logger.record_tabular("KTO negative reward", neg)
                    loss_graph.append(loss)

        elif is_DPO:
            if method == "strange_DPO":
                logger.record_tabular("strange DPO loss", loss.item())
                logger.record_tabular("strange DPO margin", margin)
                logger.record_tabular("strange DPO positive reward", pos)
                logger.record_tabular("strange DPO negative reward", neg)
                logger.record_tabular("strange DPO demonstration loss", demons)
                loss_graph.append(loss.item())
            else:
                if "warmup" in method and itr < warmup_itr:
                    logger.record_tabular("DPO warmup loss", loss.item())
                    loss_graph.append(loss.item())
                else:
                    logger.record_tabular("DPO loss", loss.item())
                    logger.record_tabular("DPO margin", margin)
                    logger.record_tabular("DPO positive reward", pos)
                    logger.record_tabular("DPO negative reward", neg)
                    loss_graph.append(loss.item())
        else:
            logger.record_tabular("BC loss", loss.item())
            loss_graph.append(loss.item())

        real_return_det, real_return_sto = try_evaluate(itr, "Running")
        # loss_graph.append(loss.item())
        real_return_det_graph.append(real_return_det)
        best_score_det = max(best_score_det, real_return_det)
        real_return_sto_graph.append(real_return_sto)
        best_score_sto = max(best_score_sto, real_return_sto)
        normalize_greedy_score.append((real_return_det-random_score)/(expert_score-random_score))
        
        normalize_stochastic_score.append((real_return_sto-random_score)/(expert_score-random_score))

        logger.record_tabular("Iteration", itr)
        logger.dump_tabular()
    plt.plot(loss_graph)
    plt.savefig(f"{graph_dir}/loss.png")
    plt.close()
    plt.plot(real_return_det_graph)
    plt.savefig(f"{graph_dir}/Greedy.png")
    plt.close()
    plt.plot(real_return_sto_graph)
    plt.savefig(f"{graph_dir}/Sample.png")
    plt.plot(normalize_greedy_score)
    plt.savefig(f"{graph_dir}/normalize_greedy.png")
    plt.close()
    plt.plot(normalize_stochastic_score)
    plt.savefig(f"{graph_dir}/normalize_stochastic.png")
    plt.close()


    if is_DPO or is_KTO:
        plt.plot(margin_graph)
        plt.savefig(f"{graph_dir}/margin.png")
        plt.close()
        plt.plot(positive_reward_graph)
        plt.savefig(f"{graph_dir}/positive_reward.png")
        plt.close()
        plt.plot(negative_reward_graph)
        plt.savefig(f"{graph_dir}/negative_reward.png")
        plt.close()
    # save data to the same csv file in graph_dir including loss, greedy, stochastic, margin, positive reward, negative reward,norm_greedy,norm_stochastic
    import csv
    # need to know which env and which method
    with open(f"{graph_dir}/{en}_{method}.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "greedy", "stochastic", "margin", "positive reward", "negative reward","norm_greedy","norm_stochastic"])
        for i in range(len(loss_graph)):
            writer.writerow([loss_graph[i],real_return_det_graph[i],real_return_sto_graph[i],margin_graph[i],positive_reward_graph[i],negative_reward_graph[i],normalize_greedy_score[i],normalize_stochastic_score[i]])







    print("Best score deterministic: ", best_score_det)
    print("Best score stochastic: ", best_score_sto)


            
    

 