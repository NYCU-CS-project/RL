#    # visualize "baselines/plt/{env_name}/exp-{num_expert_trajs}/total.csv" there will be all kinds of method,dataframe column is method,itr,loss,real_return_det,real_return_sto,margin,positive_reward,negative_reward
# # I want to visualize real_return_det,real_return_sto in the same plot, and the x-axis is itr. and the y-axis is real_return_det,real_return_sto. each method has a different color.

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys

# # also show the expert score as a horizontal line in the plot
# # AntFH-v0 expert score = 5926.18+-124.56
# # Walker2dFH-v0 expert score = 5344.21+-84.45
# # HopperFH-v0 expert score = 3592.63+-19.21
# # HalfCheetahFH-v0 expert score = 12427.49+-486.38

# # sys.argv[1] = env_name
# # sys.argv[2] = num_expert_trajs
# env_name = sys.argv[1]
# num_expert_trajs = sys.argv[2]
# path = f"plt/{env_name}/exp-{num_expert_trajs}/total.csv"
# df = pd.read_csv(path)
# methods = df['method'].unique()
# for method in methods:
#     df_method = df[df['method'] == method]
#     plt.plot(df_method['itr'], df_method['real_return_det'], label=f'{method}_greedy')
#     plt.plot(df_method['itr'], df_method['real_return_sto'], label=f'{method}_stochastic')
# if env_name == 'AntFH-v0':
#     plt.axhline(y=5926.18, color='r', linestyle='-', label='expert')
# elif env_name == 'Walker2dFH-v0':
#     plt.axhline(y=5344.21, color='r', linestyle='-', label='expert')
# elif env_name == 'HopperFH-v0':
#     plt.axhline(y=3592.63, color='r', linestyle='-', label='expert')
# elif env_name == 'HalfCheetahFH-v0':
#     plt.axhline(y=12427.49, color='r', linestyle='-', label='expert')
# plt.xlabel('itr')
# plt.ylabel('real_return')
# plt.legend()
# plt.savefig(f'plt/{env_name}/exp-{num_expert_trajs}/result.png')

# # # no pandas ver
# # import os
# # import matplotlib.pyplot as plt
# # import sys

# # # also show the expert score as a horizontal line in the plot

# # env_name = sys.argv[1]
# # num_expert_trajs = sys.argv[2]
# # path = f"plt/{env_name}/exp-{num_expert_trajs}/total.csv"
# # with open(path) as f:
# #     lines = f.readlines()
# #     methods = []
# #     for line in lines:
# #         method = line.split(',')[0]
# #         if method not in methods:
# #             methods.append(method)
# #     for method in methods:
# #         itr = []
# #         real_return_det = []
# #         real_return_sto = []
# #         for line in lines:
# #             if method in line:
# #                 itr.append(line.split(',')[1])
# #                 real_return_det.append(line.split(',')[3])
# #                 real_return_sto.append(line.split(',')[4])
# #         plt.plot(itr, real_return_det, label=f'{method}_greedy')
# #         plt.plot(itr, real_return_sto, label=f'{method}_stochastic')
# #     if env_name == 'AntFH-v0':
# #         plt.axhline(y=5926.18, color='r', linestyle='-', label='expert')
# #     elif env_name == 'Walker2dFH-v0':
# #         plt.axhline(y=5344.21, color='r', linestyle='-', label='expert')
# #     elif env_name == 'HopperFH-v0':
# #         plt.axhline(y=3592.63, color='r', linestyle='-', label='expert')
# #     elif env_name == 'HalfCheetahFH-v0':
# #         plt.axhline(y=12427.49, color='r', linestyle='-', label='expert')
# #     plt.xlabel('itr')
# #     plt.ylabel('score')
# #     plt.legend()
# #     plt.savefig(f'plt/{env_name}/exp-{num_expert_trajs}/result.png')

import gymnasium as gym
import numpy as np

def random_policy(observation, action_space):
    return action_space.sample()

def evaluate_policy(env, policy, n_episodes=10):
    episode_rewards = []
    for _ in range(n_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = policy(observation, env.action_space)
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

if __name__ == "__main__":
    env_name = "Ant-v2"  # Using v4 as it's the latest version in Gymnasium
    env = gym.make(env_name)
    
    print(f"Testing random policy on {env_name}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    mean_reward, std_reward = evaluate_policy(env, random_policy)
    
    print(f"Mean episode reward: {mean_reward:.2f}")
    print(f"Std of episode reward: {std_reward:.2f}")

    env.close()
