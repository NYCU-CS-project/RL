

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "CartPole-v1"
model_path = "/mnt/nfs/work/c98181/rl-baselines3-zoo/rl-trained-agents/ppo/CartPole-v1_1/CartPole-v1.zip"  # 模型文件的路径
# env_id = "MsPacmanNoFrameskip-v4"
# model_path = "/mnt/nfs/work/c98181/rl-baselines3-zoo/rl-trained-agents/dqn/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4.zip" # 模型文件的路径

from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.preprocessing import is_image_space
env = make_vec_env(env_id, n_envs=1)
# env = make_atari_env(env_id, n_envs=1)
# env = VecFrameStack(env, n_stack=4)

# # 如果环境的观测空间是图像，则转换图像的通道顺序
# if is_image_space(env.observation_space):
#     env = VecTransposeImage(env)

model = PPO.load(model_path, env=env)
# print model size
# print(model.policy)


import numpy as np
from tqdm import tqdm
obs = env.reset()
# initial_obs = obs
# print(initial_obs)
done = False
# sample 1M steps for CartPole and save to a numpy file
n_trajectories = 5
# n_steps = 28000
obs_list = []
next_obs_list = []
actions_list = []
rewards_list = []
dones_list = []
info_list = []
score_list=[]
reward_sum = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    next_obs, rewards, dones, info = env.step(action)
    
    reward_sum += rewards[0]
    obs_list.append(obs)
    next_obs_list.append(next_obs)
    actions_list.append(action)
    rewards_list.append(rewards)
    dones_list.append(dones)
    info_list.append(info)
    obs = next_obs
    if dones[0]:
        score_list.append(reward_sum)
        reward_sum = 0
        obs = env.reset()
        n_trajectories -= 1
        if n_trajectories == 0:
            break

obs_list = np.array(obs_list)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)
dones_list = np.array(dones_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+f"_28000_obs.npy", obs_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_actions.npy", actions_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_rewards.npy", rewards_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_dones.npy", dones_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_info.npy", info_list)
# np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_next_obs.npy", next_obs_list)
print(obs_list.shape)
print(actions_list.shape)
print(rewards_list.shape)
print(dones_list.shape)
# print(info_list)
print(next_obs_list.shape)
# print(score_list)
#　plot the score
import matplotlib.pyplot as plt
plt.plot(score_list)
plt.xlabel("Episodes")

plt.ylabel("Rewards")
plt.title("Rewards of DQN on "+env_id)
plt.savefig("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_28000_rewards.png")
plt.show()

# save to /mnt/nfs/work/c98181/cfil/CFIL/expert_datasets/spinningup_data/Humanoid-v2.npz
# format is ['states', 'actions', 'next_states', 'dones', 'rewards']
