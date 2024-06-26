
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
    "seals:seals/Humanoid-v1",
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
    env_name="seals:seals/Humanoid-v1",
    venv=env,
)
print(model)
# rollout_info = rollout.generate_trajectories(model, env,rollout.make_sample_until(min_timesteps=None, min_episodes=1),rng=np.random.default_rng())
# # print(rollout_info[0])
# transitions = rollout.flatten_trajectories(rollout_info)

# print(transitions.obs.shape)
import numpy as np
from tqdm import tqdm
obs = env.reset()
done = False
# n_steps = 10000000
num_episodes = 5

obs_list = []
next_obs_list = []
actions_list = []
rewards_list = []
dones_list = []
info_list = []
score_list=[]
reward_sum = 0
for i in tqdm(range(num_episodes)):
    obs = env.reset()

    while(1):
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
            score_list.append(reward_sum)
            reward_sum = 0
            break

obs=np.array(obs_list)
acts=np.array(actions_list)

np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), obs)
np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), acts)
# np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), transitions.obs)
# np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), transitions.acts)
# np.save(os.path.join(os.path.dirname(__file__),"dones.npy"), transitions.dones)
# np.save(os.path.join(os.path.dirname(__file__),"info.npy"), transitions.infos)
# np.save(os.path.join(os.path.dirname(__file__),"next_obs.npy"), transitions.next_obs)

import matplotlib.pyplot as plt
# show score

plt.plot(score_list)
plt.savefig(os.path.join(os.path.dirname(__file__),"score.png"))
# print shape
print(obs.shape)
print(acts.shape)
print(np.array(rewards_list).shape)
print(np.array(dones_list).shape)
print(np.array(info_list).shape)
print(np.array(next_obs_list).shape)

np.savez("/mnt/nfs/work/c98181/cfil/CFIL/expert_datasets/spinningup_data/Humanoid-v2.npz", states=obs_list, actions=actions_list, next_states=next_obs_list, dones=dones_list, rewards=rewards_list)
