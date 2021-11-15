import gym
import numpy as np
import json

from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, TD3
import SimulationEnvs
from tensorboard_logger import TensorboardCallback
import matplotlib.pyplot as plt


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# create DmsSim Gym Env
env = SimulationEnvs.NoControllerAdaptivePT2()

# use action and param noise?
n_actions = env.action_space.shape[-1]
# n_actions = 2
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.005) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.15) * np.ones(n_actions))


# # use DDPG and create a tensorboard
# # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
# policy_kwargs = dict(net_arch=[128, 128])
model = DDPG(MlpPolicy, env, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/")#, policy_kwargs=policy_kwargs)
# model = TD3("MlpPolicy", env, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/", batch_size=300, gamma=0.1)#, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=30000, tb_log_name="first_run", callback=TensorboardCallback(env))

# save model if you want to
model.save("test_save")
#

# # load model if you want to
# model = DDPG.load("test_save")

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
    ob = env.reset()
    dones = False
    rewards = []
    obs = []
    actions = []
    while dones is not True:
        action, _states = model.predict(ob)
        actions.append(action)
        print(f"Action {action}")
        ob, reward, dones, info = env.step(action)
        print(f"Observation: {ob}")
        print(f"Reward {reward}")
        rewards.append(reward)
        obs.append(ob)
    obs = np.array(obs)
    print(np.mean(obs[:, 0]))
    # print(np.mean(obs[:, 1]))
    plt.plot(obs)
    plt.show()
    plt.plot(actions)
    plt.show()
    plt.plot(rewards)
    plt.show()
    plt.plot(env.u)
    plt.plot(env.out)
    plt.show()
    print(np.mean(rewards))
    exit()
    env.render()