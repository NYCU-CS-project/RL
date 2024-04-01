# import gymnasium as gym

# from stable_baselines3 import DQN
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


# env_id = "CartPole-v1"
# # Create environment
# env = gym.make(env_id, render_mode="rgb_array")

# # Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)
# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(2e5), progress_bar=True)
# # Save the agent
# model.save(env_id)
# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DQN.load(env_id, env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# # # Enjoy trained agent
# # vec_env = model.get_env()
# # obs = vec_env.reset()
# # for i in range(1000):
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, rewards, dones, info = vec_env.step(action)
# #     vec_env.render("human")


# video_folder = "logs/videos/"
# video_length = 1000

# vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

# obs = vec_env.reset()

# # Record the video starting at the first step
# vec_env = VecVideoRecorder(vec_env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix=f"{env_id}")

# vec_env.reset()
# for _ in range(video_length + 1):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
# # Save the video
# vec_env.close()

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env,make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "CartPole-v1"
model_path = "/mnt/nfs/work/c98181/rl-baselines3-zoo/rl-trained-agents/dqn/CartPole-v1_1/CartPole-v1.zip"  # 模型文件的路径
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

model = DQN.load(model_path, env=env)
# print model size
# print(model.policy)


import numpy as np
from tqdm import tqdm
obs = env.reset()
# initial_obs = obs
# print(initial_obs)
done = False
# sample 1M steps for CartPole and save to a numpy file
n_steps = 1000000
obs_list = []
actions_list = []
rewards_list = []
dones_list = []
score_list=[]
reward_sum = 0
for _ in tqdm(range(n_steps)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    obs_list.append(obs)
    actions_list.append(action)
    rewards_list.append(reward)
    dones_list.append(done)
    reward_sum += reward
    if done:
        score_list.append(reward_sum)
        reward_sum = 0
        obs = env.reset()
obs_list = np.array(obs_list)
actions_list = np.array(actions_list)
rewards_list = np.array(rewards_list)
dones_list = np.array(dones_list)
np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_1M_obs.npy", obs_list)
np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_1M_actions.npy", actions_list)
np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_1M_rewards.npy", rewards_list)
np.save("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_1M_dones.npy", dones_list)

#　plot the score
import matplotlib.pyplot as plt
plt.plot(score_list)
plt.xlabel("Episodes")

plt.ylabel("Rewards")
plt.title("Rewards of DQN on "+env_id)
plt.savefig("/mnt/nfs/work/c98181/RL/dataset/"+env_id+"_1M_rewards.png")
plt.show()