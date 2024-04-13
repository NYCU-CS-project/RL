from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os
env_id = "MsPacmanNoFrameskip-v4"
# get the current directory

model_path = os.path.join(os.path.dirname(__file__), env_id+".zip")
print(model_path)

from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.preprocessing import is_image_space
env = make_atari_env(env_id, n_envs=1)
env=VecFrameStack(env, n_stack=4)

model = DQN.load(model_path, env=env)
model.policy.to("cuda")
print(model.policy)

import numpy as np
from tqdm import tqdm
obs = env.reset()
done = False
# sample 10M steps
n_steps = 10000000
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
