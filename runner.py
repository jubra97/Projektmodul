import gym
import numpy as np
import json

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, TD3
import SimulationEnvs
from tensorboard_logger import TensorboardCallback
import matplotlib.pyplot as plt

# create DmsSim Gym Env
env = SimulationEnvs.NoControllerAdaptivePT2()

# use action and param noise?
n_actions = env.action_space.shape[-1]
# n_actions = 2
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.25) * np.ones(n_actions))
# noise = []
# noise2 = []
# for i in range(10000):
#     # if i % 250 == 0:
#     #     action_noise.reset()
#     noise.append(action_noise())
# noise = np.array(noise)
# noise2 = np.array(noise2)
# plt.plot(noise)
# plt.show()
# # use DDPG and create a tensorboard
# # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
# policy_kwargs = dict(net_arch=[10, 10])
model = DDPG(MlpPolicy, env, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/")#, policy_kwargs=policy_kwargs)
#
# model = TD3("MlpPolicy", env, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/", batch_size=300, gamma=0.1)#, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=50000, tb_log_name="first_run", callback=TensorboardCallback(env))
# # # #
# # # # save model if you want to
# model.save("test_save")
# #
# # del model # remove to demonstrate saving and loading
#
# # load model if you want to
model = DDPG.load("test_save")
#
# # simulate solutions
# plt.plot(np.array(env.rewards_log), label="Rewards")
# # plt.plot(np.array(env.actions_log), label="Actions")
# # plt.plot(np.array(env.observations_log), label="Observations")
# plt.plot(np.array(env.dones_log), label="Dones")
# plt.legend()
# plt.grid()
# plt.show()
save_dict = {"obs:": env.observations_log,
             "actions:": env.actions_log,
             "rewards": env.rewards_log,
             "done": env.dones_log}
with open("log_data_last_run..json2", "w") as f:
    json.dump(save_dict, f, indent=4)

while True:
    # env.init_render()
    obs = env.reset()
    dones = False
    rewards = []
    while dones is not True:
        action, _states = model.predict(obs)
        print(f"Action {action}")
        obs, reward, dones, info = env.step(action)
        print(f"Observation: {obs}")
        print(f"Reward {reward}")
        rewards.append(reward)
    # plt.plot(rewards)
    # plt.show()
    # plt.plot(env.u)
    # plt.plot(env.out)
    # plt.show()
    # print("A")
    env.render()