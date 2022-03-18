import numpy as np
from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList
from CustomEvalCallback import CustomEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from envs.DirectControllerOnline import DirectControllerOnline
import custom_policy_no_bias

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


for actor_net in [[20, 20]]:
    for critic_net in [[200, 200]]:
        for af, af_name in zip([th.nn.Tanh], ["TanH"]):


            online_connection = DirectControllerOnlineConnection()
            # for action_noise in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0001, 0]:
            RUN_NAME = f"direct_with_error"


            env = make_vec_env(DirectControllerOnline, env_kwargs={"online_sys": online_connection})


            # create DmsSim Gym Env
            # env = PIControllerOnline
            # # env = Monitor(env)
            # env = DummyVecEnv([env])
            env = VecCheckNan(env)

            online_eval_env = DirectControllerOnline(online_connection, log=True)
            online_eval_env = Monitor(online_eval_env)

            # create action noise
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.005) * np.ones(n_actions))

            # create eval callback
            eval_callback = CustomEvalCallback(online_eval_env, best_model_save_path=f"eval\\{RUN_NAME}\\best_model", eval_freq=1500, deterministic=True)

            # create callback list
            callbacks = CallbackList([eval_callback])

            # # use DDPG and create a tensorboard
            # # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
            # policy_kwargs = dict(activation_fn=th.nn.Tanh)
            policy_kwargs = dict(net_arch=dict(pi=actor_net, qf=critic_net), activation_fn=af)

            model = DDPG("CustomTD3Policy",
                         env,
                         learning_starts=6000,
                         verbose=2,
                         action_noise=action_noise,
                         tensorboard_log="./direct_with_error/",
                         policy_kwargs=policy_kwargs,
                         )

            # model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\eval\direct_with_error\end_model2.zip", env, force_reset=True,
            #                   custom_objects={"learning_starts": 0,
            #                                   "action_noise": action_noise})

            model.learn(total_timesteps=50_000, tb_log_name=f"RUN", callback=callbacks, log_interval=1)
            # # utils.eval(PIControllerOnline(log=True), model, folder_name=RUN_NAME)
            # # #
            # # # # save model if you want to
            model.save(f"eval\\{RUN_NAME}\\end_model2")