import time

import numpy as np
import json
# import matplotlib
from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG, TD3
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

def linear_schedule(initial_value: float=1e-3) -> Callable[[float], float]:
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


# ACTOR_NET = [10, 10]
# CRITIC_NET = [200, 200]
# AF = th.nn.Sigmoid
# AF_STRING = "Sigmoid"


for actor_net in [[2, 2], [3, 3], [5, 5], [50, 50]]:
    for critic_net in [[100, 100], [200, 200], [300, 300], [500, 500]]:
        for af, af_name in zip([th.nn.Sigmoid, th.nn.Tanh, th.nn.ReLU], ["Sigmoid", "TanH", "ReLU"]):

            # for action_noise in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0001, 0]:
            RUN_NAME = f"obs_with_vel_diff-actor_net_{actor_net}_critic_net_{critic_net}_activation_fn_{af_name}_test"

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
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

            # create eval callback
            eval_callback = CustomEvalCallback(online_eval_env, eval_freq=5000, deterministic=True)

            # create callback list
            callbacks = CallbackList([eval_callback])

            # # use DDPG and create a tensorboard
            # # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
            # policy_kwargs = dict(activation_fn=th.nn.Tanh)
            policy_kwargs = dict(net_arch=dict(pi=actor_net, qf=critic_net), activation_fn=af)

            model = DDPG(MlpPolicy,
                         env,
                         learning_starts=3000,
                         verbose=2,
                         action_noise=action_noise,
                         tensorboard_log="./net_params_discrete_reward/",
                         policy_kwargs=policy_kwargs,
                         )
            model.learn(total_timesteps=100_000, tb_log_name=f"actor_net{actor_net}_critic_net{critic_net}_af_{af_name}", callback=callbacks)
            utils.eval(DirectControllerPT2(log=True), model, folder_name=RUN_NAME)
            # #
            # # # save model if you want to
            model.save(f"eval\\{RUN_NAME}\\model")