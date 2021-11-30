import time

import numpy as np
import json
# import matplotlib
from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
# from envs.DirectControllerv2 import DirectControllerPT2
from envs.DirectControllerv2 import DirectControllerPT2
from envs.PIAdaptivePT2 import PIAdaptivePT2
from tensorboard_logger import TensorboardCallback
import matplotlib.pyplot as plt
from CustomEvalCallback import CustomEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv
import torch as th
import utils

# def linear_schedule(initial_value: float) -> Callable[[float], float]:
#     """
#     Linear learning rate schedule.
#
#     :param initial_value: Initial learning rate.
#     :return: schedule that computes
#       current learning rate depending on remaining progress
#     """
#
#     def func(progress_remaining: float) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0.
#
#         :param progress_remaining:
#         :return: current learning rate
#         """
#         return progress_remaining * initial_value
#
#     return func


# create DmsSim Gym Env
env = DirectControllerPT2()
env = Monitor(env)

online_eval_env = DirectControllerPT2(log=True)
online_eval_env = Monitor(online_eval_env)
# env = VecCheckNan(env)

# env = DummyVecEnv([lambda: DirectControllerPT2()])
# env = VecCheckNan(env, raise_exception=True)

# create action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.15) * np.ones(n_actions))

# create eval callback
eval_callback = CustomEvalCallback(online_eval_env, eval_freq=1500, deterministic=True)

# create callback list
callbacks = CallbackList([eval_callback])

# # use DDPG and create a tensorboard
# # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
# policy_kwargs = dict(activation_fn=th.nn.Sigmoid)
model = DDPG(MlpPolicy, env, learning_starts=3000, verbose=2, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/")#, policy_kwargs=policy_kwargs,)
model.learn(total_timesteps=50000, tb_log_name="direct_control", callback=callbacks)
utils.eval(DirectControllerPT2(log=True), model)
# #
# # # save model if you want to
# model.save("direct_control")
# model.save_replay_buffer("replay_buffer")
# #

# # load model if you want to
# model = DDPG.load("direct_control.zip")
# utils.eval(DirectControllerPT2, model)

# while True:
#     # env.init_render()
#     ob = env.reset()
#     dones = False
#     rewards = []
#     obs = []
#     actions = []
#     while dones is not True:
#         action, _states = model.predict(ob)
#         actions.append(action)
#         print(f"Action {action}")
#         ob, reward, dones, info = env.step(action)
#         print(f"Observation: {ob}")
#         print(f"Reward {reward}")
#         rewards.append(reward)
#         obs.append(ob)
#     obs = np.array(obs)
#     print(np.mean(obs[:, 0]))
#     # print(np.mean(obs[:, 1]))
#     # matplotlib.use("TkAgg")
#     # plt.plot(obs)
#     # plt.show()
#     # plt.plot(actions)
#     # plt.show()
#     # plt.plot(rewards)
#     # plt.show()
#     # plt.plot(env.u)
#     # plt.plot(env.out)
#     # plt.show()
#     print(np.mean(rewards))
#     # exit()
#     env.render()
#     time.sleep(0.1)
