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
from envs.PIAdaptivePT2 import PIAdaptivePT2
from tensorboard_logger import TensorboardCallback
import matplotlib.pyplot as plt
from CustomEvalCallback import CustomEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import utils
# from envs.DirectControllerOnline import DirectControllerOnline

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

if __name__ == "__main__":
    for actor_net in [[20, 20]]:
        for critic_net in [[200, 200]]:
            for af, af_name in zip([th.nn.Tanh], ["TanH"]):

                # for action_noise in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0001, 0]:
                RUN_NAME = f"obs_with_vel_diff-actor_net_{actor_net}_critic_net_{critic_net}_activation_fn_{af_name}_test"

                # create DmsSim Gym Env
                # env = PIAdaptivePT2()
                # env = Monitor(env)

                env = make_vec_env(PIAdaptivePT2, 2, vec_env_cls=SubprocVecEnv)

                online_eval_env = PIAdaptivePT2(log=True)
                online_eval_env = Monitor(online_eval_env)

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
                             tensorboard_log="./test/",
                             policy_kwargs=policy_kwargs,
                             train_freq=1,
                             gradient_steps=1
                             )
                model.learn(total_timesteps=50_000, tb_log_name=f"test", callback=callbacks)
                utils.eval(PIAdaptivePT2(log=True), model, folder_name=RUN_NAME)
                # #
                # # # save model if you want to
                model.save(f"eval\\{RUN_NAME}\\model")