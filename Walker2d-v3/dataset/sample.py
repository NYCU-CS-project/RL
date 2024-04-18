
# from stable_baselines3 import DQN, PPO
# from stable_baselines3.common.env_util import make_vec_env,make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# import os
# import stable_baselines3.common.env_util
# env_id = "Walker2d-v4"
# # get the current directory

# # model_path = os.path.join(os.path.dirname(__file__), env_id+".zip")
# # print(model_path)
# import torch
# from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
# from stable_baselines3.common.preprocessing import is_image_space
# env = make_vec_env(env_id, n_envs=1)

# model = PPO.load(os.path.join(os.path.dirname(__file__),"Walker2d-v3"+".zip"), env=env)



# print(model.policy)

# import numpy as np
# from tqdm import tqdm
# obs = env.reset()
# done = False
# # sample 1M steps
# n_steps = 100000
# # num_episodes = 100

# obs_list = []
# next_obs_list = []
# actions_list = []
# rewards_list = []
# dones_list = []
# info_list = []
# score_list=[]
# reward_sum = 0
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

# obs_list = np.array(obs_list)
# actions_list = np.array(actions_list)
# rewards_list = np.array(rewards_list)
# dones_list = np.array(dones_list)
# np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), obs_list)
# np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), actions_list)
# np.save(os.path.join(os.path.dirname(__file__),"rewards.npy"), rewards_list)
# np.save(os.path.join(os.path.dirname(__file__),"dones.npy"), dones_list)
# np.save(os.path.join(os.path.dirname(__file__),"info.npy"), info_list)
# np.save(os.path.join(os.path.dirname(__file__),"next_obs.npy"), next_obs_list)

# #　plot the score
# import matplotlib.pyplot as plt
# plt.plot(score_list)
# plt.xlabel("Episodes")

# plt.ylabel("Rewards")
# plt.title("Rewards of sample on "+env_id)
# plt.savefig(os.path.join(os.path.dirname(__file__),"rewards.png"))
# plt.show()

# # import agc.dataset as ds
# # import agc.util as util

# # # DATA_DIR is the directory, which contains the 'trajectories' and 'screens' folders
# # dataset = ds.AtariDataset(DATA_DIR)


# # # dataset.trajectories returns the dictionary with all the trajs from the dataset
# # all_trajectories = dataset.trajectories
# import gymnasium as gym
# import matplotlib.pyplot as plt

# env = gym.make('Walker2d-v4')  # 確保使用正確的環境ID
# env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)  # Gymnasium 可能支持'truncated'返回
#     frame = env.render(mode="rgb_array")
#     print(frame)
# env.close()
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout, types as rtypes
env = make_vec_env(
    "seals:seals/Walker2d-v1",
    rng=np.random.default_rng(),
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals/Walker2d-v1",
    venv=env,
)
rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=10),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)
print(transitions)
