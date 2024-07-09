
# import numpy as np
# import gymnasium as gym
# from imitation.policies.serialize import load_policy
# from imitation.util.util import make_vec_env
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.data import rollout, types as rtypes
# import os
# import torch
# # env=gym.make("Walker2d-v3")
# # state=env.reset()
# # print(env.step([1,1,1,1,1,1]))
# # test env
# env = make_vec_env(
#     "seals:seals/HalfCheetah-v1",
#     rng=np.random.default_rng(),
#     post_wrappers=[
#         lambda env, _: RolloutInfoWrapper(env)
#     ],  # needed for computing rollouts later
#     env_make_kwargs={"render_mode": "rgb_array"},
#     n_envs=1,
# )
# model = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals/Walker2d-v1",
#     venv=env,
# )
# # rollout_info = rollout.generate_trajectories(model, env,rollout.make_sample_until(min_timesteps=None, min_episodes=1),rng=np.random.default_rng())
# # # print(rollout_info[0])
# # transitions = rollout.flatten_trajectories(rollout_info)

# # print(transitions.obs.shape)
# import numpy as np
# from tqdm import tqdm
# obs = env.reset()
# done = False
# # n_steps = 10000000
# num_episodes = 5

# obs_list = []
# next_obs_list = []
# actions_list = []
# rewards_list = []
# dones_list = []
# info_list = []
# score_list=[]
# reward_sum = 0
# for i in tqdm(range(num_episodes)):
#     obs = env.reset()

#     while(1):
#         action, _states = model.predict(obs, deterministic=True)
#         next_obs, reward, done, info = env.step(action)
#         reward_sum += reward
#         obs_list.append(obs)
#         next_obs_list.append(next_obs)
#         actions_list.append(action)
#         rewards_list.append(reward)
#         dones_list.append(done)
#         info_list.append(info)
#         obs = next_obs
#         if done:
#             score_list.append(reward_sum)
#             reward_sum = 0
#             break

# obs=np.array(obs_list)
# acts=np.array(actions_list)

# np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), obs)
# np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), acts)
# # np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), transitions.obs)
# # np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), transitions.acts)
# # np.save(os.path.join(os.path.dirname(__file__),"dones.npy"), transitions.dones)
# # np.save(os.path.join(os.path.dirname(__file__),"info.npy"), transitions.infos)
# # np.save(os.path.join(os.path.dirname(__file__),"next_obs.npy"), transitions.next_obs)

# import matplotlib.pyplot as plt
# # show score

# plt.plot(score_list)
# plt.savefig(os.path.join(os.path.dirname(__file__),"score.png"))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import pandas as pd
import os
SEED = 42
result=[]

env = make_vec_env(
    "seals:seals/HalfCheetah-v1",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)

# Load expert policy
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals-HalfCheetah-v1",
    venv=env,
)

# Generate expert rollouts
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    rng=np.random.default_rng(SEED),
)

for i in range(5):
    print(len(rollouts[i].obs))
#flatten_trajectories return a named tuple with fields obs, acts, dones, infos, next_obs
# save all the obs and observe in rollouts to a file
flat=rollout.flatten_trajectories(rollouts)
print(flat.obs.shape)
# save obs and actions at this direxcotry with os.path.join(os.path.dirname(__file__))
# np.save("obs.npy", flat.obs)
# np.save("actions.npy", flat.acts)
np.save(os.path.join(os.path.dirname(__file__),"obs.npy"), flat.obs)
np.save(os.path.join(os.path.dirname(__file__),"actions.npy"), flat.acts)


# Initialize learner
learner = PPO(
    env=env,
    policy='MlpPolicy',
    batch_size=32,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=1,
    seed=SEED,
)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=512,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# # Evaluate the learner before training
env.seed(SEED)
# learner_rewards_before_training, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)

# eval_callback = CustomEvalCallback(env, eval_freq=5000)

# def combined_callback(round_num):
#     logs = {
#         'gen_algo': gail_trainer.gen_algo,
#         'loss': gail_trainer.gen_algo.logger.name_to_value,  # Correct method to get log data
#     }
#     # logging_callback(round_num, logs)
#     eval_callback(round_num, logs)

num_epochs = 40
# gail_trainer.train(5000 * num_epochs, callback=combined_callback)
for epoch in range(num_epochs):
    # gail_trainer.train(20000, callback=combined_callback)
    gail_trainer.train(10000)
    rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
    result.append(np.mean(rewards))
# learner_rewards, _ = evaluate_policy(learner, env, 10, return_episode_rewards=True)
# print("Learner rewards after training:", learner_rewards)
name="HalfCheetah-v1"
# Save rewards and losses to CSV files at the same directory by os.path.join(os.path.dirname(__file__))
rewards_df = pd.DataFrame(result, columns=['Reward'])
# rewards_df.to_csv('epoch_rewards.csv', index=False)
rewards_df.to_csv(os.path.join(os.path.dirname(__file__),"epoch_rewards.csv"), index=False)
print(len(result))

# actor_losses_df = pd.DataFrame(logging_callback.actor_losses, columns=['Actor Loss'])
# actor_losses_df.to_csv(f'{name}_actor_losses.csv', index=False)

# critic_losses_df = pd.DataFrame(logging_callback.critic_losses, columns=['Critic Loss'])
# critic_losses_df.to_csv(f'{name}_critic_losses.csv', index=False)

print("Rewards per epoch:", result)

