from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "MountainCar-v0"
model_path = "/mnt/nfs/work/c98181/rl-baselines3-zoo/rl-trained-agents/ppo/MountainCar-v0_1/MountainCar-v0.zip"  # 模型文件的路径
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
n_steps = 28000
obs_list = []
next_obs_list = []
actions_list = []
rewards_list = []
dones_list = []
info_list = []
score_list=[]
reward_sum = 0

for i in tqdm(range(n_steps)):
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
    
obs_list = np.array(obs_list)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)
dones_list = np.array(dones_list)
# check shape 
print(obs_list.shape, actions_list.shape, rewards_list.shape, dones_list.shape)
print(obs_list[0], actions_list[0], rewards_list[0], dones_list[0])
# /mnt/nfs/work/c98181/RL/MountainCar-v0/dataset
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_obs.npy", obs_list)
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_actions.npy", actions_list)
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_rewards.npy", rewards_list)
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_dones.npy", dones_list)
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_info.npy", info_list)
np.save("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_next_obs.npy", next_obs_list)

#　plot the score
import matplotlib.pyplot as plt
plt.plot(score_list)
plt.xlabel("Episodes")

plt.ylabel("Rewards")
plt.title("Rewards of DQN on "+env_id)
# dataset/MountainCar-v0_28000_rewards.png
plt.savefig("/mnt/nfs/work/c98181/RL/MountainCar-v0/dataset/MountainCar-v0_28000_rewards.png")
plt.show()
