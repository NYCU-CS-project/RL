
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout, types as rtypes
import os
import torch
# env=gym.make("Walker2d-v3")
# state=env.reset()
# print(env.step([1,1,1,1,1,1]))
# test env
env = make_vec_env(
    "seals:seals/Ant-v1",
    rng=np.random.default_rng(),
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
    env_make_kwargs={"render_mode": "rgb_array"},
    n_envs=1,
)
model = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals/Ant-v1",
    venv=env,
)
print(model)
import numpy as np
from tqdm import tqdm
obs = env.reset()
done = False
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
plt.title("Rewards of sample on walker2d")
plt.savefig(os.path.join(os.path.dirname(__file__),"rewards.png"))
plt.show()
# import gym
# # load monitor
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
# env=gym.make("Ant-v3")
# #test 
# import os
# env.reset()
# obs, reward, done, info = env.step(env.action_space.sample())
# x=env.render("rgb_array")
# print(x.shape)
# import matplotlib.pyplot as plt
# plt.imshow(x)
# plt.savefig(os.path.join(os.path.dirname(__file__),"test.png"))