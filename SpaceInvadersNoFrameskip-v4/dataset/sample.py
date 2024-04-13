from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os
env_id = "SpaceInvadersNoFrameskip-v4"
# get the current directory

# model_path = os.path.join(os.path.dirname(__file__), env_id+".zip")
# print(model_path)
import torch
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.preprocessing import is_image_space
env = make_atari_env(env_id, n_envs=1)
env = VecTransposeImage(env)            # 确保图像通道在前

env=VecFrameStack(env, n_stack=4)

model = DQN("CnnPolicy", env, optimize_memory_usage=False, buffer_size=10)
model.policy.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "policy.pth")))



print(model.policy)

import numpy as np
from tqdm import tqdm
obs = env.reset()
done = False
# sample 1M steps
n_steps = 100000
# num_episodes = 100

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
    next_obs, reward, done, info = env.step(action)
    reward_sum += reward
    obs_list.append(obs)
    next_obs_list.append(next_obs)
    actions_list.append(action)
    rewards_list.append(reward)
    dones_list.append(done)
    info_list.append(info)
    obs = next_obs
    if done:
        obs = env.reset()
        score_list.append(reward_sum)
        reward_sum = 0

obs_list = np.array(obs_list)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)
dones_list = np.array(dones_list)
np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), obs_list)
np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), actions_list)
np.save(os.path.join(os.path.dirname(__file__),"rewards.npy"), rewards_list)
np.save(os.path.join(os.path.dirname(__file__),"dones.npy"), dones_list)
np.save(os.path.join(os.path.dirname(__file__),"info.npy"), info_list)
np.save(os.path.join(os.path.dirname(__file__),"next_obs.npy"), next_obs_list)

#　plot the score
import matplotlib.pyplot as plt
plt.plot(score_list)
plt.xlabel("Episodes")

plt.ylabel("Rewards")
plt.title("Rewards of sample on "+env_id)
plt.savefig(os.path.join(os.path.dirname(__file__),"rewards.png"))
plt.show()

# import agc.dataset as ds
# import agc.util as util

# # DATA_DIR is the directory, which contains the 'trajectories' and 'screens' folders
# dataset = ds.AtariDataset(DATA_DIR)


# # dataset.trajectories returns the dictionary with all the trajs from the dataset
# all_trajectories = dataset.trajectories
