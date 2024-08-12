# '''
# Behavior cloning MLE(Learnt variance) and (MSE)Fixed variance policy.
# '''

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML

from common.sac import ReplayBuffer, SAC
import matplotlib.pyplot as plt
import envs
from utils import system, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy

def try_evaluate(itr: int, policy_type: str):
    assert policy_type in ["Running"]
    update_time = itr * v['bc']['eval_freq']


    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], True)

    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], False)

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
            log_pi = sac_agent.ac.pi.log_prob(torch.FloatTensor(expert_states[start_id:end_id,:]),\
                                                            torch.FloatTensor(expert_actions[start_id:end_id,:]))
            sac_agent.pi_optimizer.zero_grad()
            nll = -(log_pi).mean()
            total_loss+=-(log_pi).sum()
            nll.backward()
            sac_agent.pi_optimizer.step()

    total_loss = total_loss/(epochs*expert_states.shape[0])

    return total_loss               
def DPO(sac_agent, prev_model, expert_states, expert_actions, greedy=False,epochs=100):
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
            losses = -torch.nn.functional.logsigmoid(beta * logits)
            loss = losses.mean()
            # check nan
            if torch.isnan(loss).any():
                print("nan in loss")
            total_loss += loss.sum()

            sac_agent.pi_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(sac_agent.ac.pi.parameters(), max_norm=1.0)
            sac_agent.pi_optimizer.step()

        prev_model.load_state_dict(sac_agent.ac.pi.state_dict())

    total_loss = total_loss / (epochs * expert_states.shape[0])
    return total_loss, total_margin, total_positive_reward, total_negative_reward


def mse_bc(sac_agent, expert_states, expert_actions, epochs = 100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 1000
    total_loss = 0
    for i in range(epochs):
        for batch_no in range(expert_states.shape[0]//batch_size):
            start_id = batch_no*batch_size
            end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
            se = ((sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]))[0] - torch.FloatTensor(expert_actions[start_id:end_id,:]))**2).sum(1)
            loss = se.mean()
            sac_agent.pi_optimizer.zero_grad()
            total_loss+=se.sum()
            loss.backward()
            sac_agent.pi_optimizer.step()

    total_loss = total_loss/(epochs*expert_states.shape[0])

    return total_loss               
    




if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['bc']['expert_episodes']

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
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_state_trajs = torch.load(f'expert_data/states/{env_name}_airl.pt').numpy()[:, :, state_indices]
    expert_state_trajs = expert_state_trajs[:num_expert_trajs, :-1, :] # select first expert_episodes
    expert_states = expert_state_trajs.copy().reshape(-1, len(state_indices))
    print(expert_state_trajs.shape, expert_states.shape) # ignored starting state

    expert_action_trajs = torch.load(f'expert_data/actions/{env_name}_airl.pt').numpy()
    expert_action_trajs = expert_action_trajs[:num_expert_trajs, 1:, :] # select first expert_episodes
    expert_actions = expert_action_trajs.reshape(-1, gym_env.action_space.shape[0])
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
    method = "DPO_stochastic"
    if method == "DPO_greedy" or method == "DPO_stochastic":
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
        

        # draw loss real return det and sto
        

        logger.record_tabular("BC loss", loss.item())

        real_return_det, real_return_sto = try_evaluate(itr, "Running")
        loss_graph.append(loss.item())
        real_return_det_graph.append(real_return_det)
        real_return_sto_graph.append(real_return_sto)
        logger.record_tabular("Iteration", itr)
        logger.dump_tabular()
    
    # draw loss real return det and sto by matplotlib
    graph_dir = f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir, exist_ok=True,mode=0o777)
    plt.plot(loss_graph)
    plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/loss.png")
    plt.close()
    plt.plot(real_return_det_graph)
    plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/Greedy.png")
    plt.close()
    plt.plot(real_return_sto_graph)
    plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/Sample.png")
    plt.close()
    if method == "DPO":
        plt.plot(margin_graph)
        plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/margin.png")
        plt.close()
        plt.plot(positive_reward_graph)
        plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/positive_reward.png")
        plt.close()
        plt.plot(negative_reward_graph)
        plt.savefig(f"baselines/plt/{env_name}/exp-{num_expert_trajs}/{method}/negative_reward.png")
        plt.close()

    

 