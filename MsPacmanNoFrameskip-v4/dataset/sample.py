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
env = make_atari_env(env_id, n_envs=1,wrapper_kwargs={"clip_reward":False})
env = VecTransposeImage(env)            # 确保图像通道在前

env=VecFrameStack(env, n_stack=4)

model = DQN.load(model_path, env=env)
model.policy.to("cuda")
print(model.policy)

import numpy as np
from tqdm import tqdm
import imageio
obs = env.reset()
done = False
# sample 1M steps
# n_steps = 100000
num_episodes = 1

obs_list = []
next_obs_list = []
actions_list = []
rewards_list = []
dones_list = []
info_list = []
score_list=[]
reward_sum = 0
# for i in tqdm(range(n_steps)):
#     action, _states = model.predict(obs, deterministic=True)
#     next_obs, reward, done, info = env.step(action)
#     reward_sum += reward
#     obs_list.append(obs)
#     next_obs_list.append(next_obs)
#     actions_list.append(action)
#     rewards_list.append(reward)
#     dones_list.append(done)
#     info_list.append(info)
#     obs = next_obs
#     if done:
#         obs = env.reset()
#         score_list.append(reward_sum)
#         reward_sum = 0
frames=[]
for i in tqdm(range(num_episodes)):
    obs = env.reset()
    
    done = False
    reward_sum = 0
    while not done:
        frame=env.render(mode='rgb_array')
        frames.append(frame)
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
    done=False
    while not done:
        frame=env.render(mode='rgb_array')
        frames.append(frame)
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
    done=False
    while not done:
        frame=env.render(mode='rgb_array')
        frames.append(frame)
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
    score_list.append(reward_sum)
    print(f"Episode {i} reward: {reward_sum}")
    print(f"Info dictionary: {info}")
    reward_sum = 0
imageio.mimsave(os.path.join(os.path.dirname(__file__),"MsPacman.gif"), frames, duration=40)

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

# # import agc.dataset as ds
# # import agc.util as util

# # # DATA_DIR is the directory, which contains the 'trajectories' and 'screens' folders
# # dataset = ds.AtariDataset(DATA_DIR)


# # dataset.trajectories returns the dictionary with all the trajs from the dataset
# all_trajectories = dataset.trajectories
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
# from stable_baselines3.common.atari_wrappers import AtariWrapper

# def make_env(env_id, seed=None):
#     def _init():
#         env = gym.make(env_id, render_mode='rgb_array')
#         env = AtariWrapper(env, frame_skip=4, terminal_on_life_loss=False, clip_reward=False)
#         return env
#     return _init

# # 环境设置
# env_id = "ALE/MsPacman-v5"  # 使用新版本的MsPacman环境
# model_path = os.path.join(os.path.dirname(__file__), "MsPacmanNoFrameskip-v4.zip")
# print(model_path)

# # 创建环境
# env = DummyVecEnv([make_env(env_id)])
# env = VecFrameStack(env, n_stack=4)
# env = VecTransposeImage(env)

# # 加载模型
# model = DQN.load(model_path, env=env)
# model.policy.to("cuda")
# # print(model.policy)

# # 采样设置
# num_episodes = 1
# # print(env)

# # print env's __dict__ to see what's inside
# # print(env.__dict__)
# # print(env.unwrapped.__dict__)
# # def find_ale(env):
# #     if hasattr(env, 'ale'):
# #         return env.ale
# #     elif hasattr(env, 'env'):
# #         return find_ale(env.env)
# #     elif hasattr(env, 'envs'):
# #         return find_ale(env.envs[0])
# #     else:
# #         return None
    

# # ale = find_ale(env.unwrapped)
# # print("------")
# # print(ale)
# # # print all func of ale which is a module of atari_py
# # print(dir(ale))
# # print(ale.getInt("score"))
# # # if ale:
# #     original_score = ale.getScore()
# #     print(f"Original game score: {original_score}")
# obs_list = []
# next_obs_list = []
# actions_list = []
# rewards_list = []
# dones_list = []
# info_list = []
# score_list = []
# import imageio
# frames=[]
# for i in tqdm(range(num_episodes)):
#     obs = env.reset()
#     done = False
#     episode_reward = 0
#     while not done:
#         # 保存图像
#         frame=env.render(mode='rgb_array')
#         frames.append(frame)
#         action, _states = model.predict(obs, deterministic=True)
#         next_obs, reward, done, info = env.step(action)
#         episode_reward += reward[0]
#         # print(env.unwrapped.envs[0].env.env.env.env.env.env.env.__dict__)
#         # print(env.unwrapped.envs[0].env.env.env.env.env.env.env.ale)

#         # print(env.unwrapped.get_wrapper_attr('buf_rews'))

        
#         # exit()
#         # print(ale.getInt("Reward"))

        
        
#         obs_list.append(obs)
#         next_obs_list.append(next_obs)
#         actions_list.append(action)
#         rewards_list.append(reward)
#         dones_list.append(done)
#         info_list.append(info)
        
#         obs = next_obs

#     print(f"Episode {i+1} reward/score: {episode_reward}")

#     # 打印info字典的内容，以便了解其结构
#     # print(f"Info dictionary: {info}")
# imageio.mimsave(os.path.join(os.path.dirname(__file__), "MsPacman.gif"), frames, duration=40)

# # 保存数据
# np.save(os.path.join(os.path.dirname(__file__), "obs.npy"), np.array(obs_list))
# np.save(os.path.join(os.path.dirname(__file__), "actions.npy"), np.array(actions_list))
# np.save(os.path.join(os.path.dirname(__file__), "rewards.npy"), np.array(rewards_list))
# np.save(os.path.join(os.path.dirname(__file__), "dones.npy"), np.array(dones_list))
# np.save(os.path.join(os.path.dirname(__file__), "info.npy"), np.array(info_list))
# np.save(os.path.join(os.path.dirname(__file__), "next_obs.npy"), np.array(next_obs_list))

# # 绘制分数图
# plt.figure(figsize=(10, 5))
# plt.plot(score_list)
# plt.xlabel("Episodes")
# plt.ylabel("Score/Reward")
# plt.title(f"Scores/Rewards of sample on {env_id}")
# plt.savefig(os.path.join(os.path.dirname(__file__), "scores.png"))
# plt.show()

# test env with random agent
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
# from stable_baselines3.common.atari_wrappers import AtariWrapper

# env=gym.make("ALE/MsPacman-v5", render_mode='rgb_array')
# obs=env.reset()
# frames=[]
# while True:
#     # frame=env.render(mode='rgb_array')
#     # frames.append(frame)
#     action=env.action_space.sample()
#     # obs, reward, done, info=env.step(action)
#     # 5-tuple with truncate,
    
#     print(reward)
#     if done:
#         break
# env.close()
# import imageio
# # imageio.mimsave(os.path.join(os.path.dirname(__file__),"random.gif"), frames, duration=40)
