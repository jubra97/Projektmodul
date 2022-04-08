import os
import pathlib
import json

import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

import custom_policy
from CustomEvalCallback import CustomEvalCallback
from envs.DirectControllerOnline import DirectControllerOnline
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
import utils

observation_options = {
    # other possible options: "raw_with_vel", "raw_with_states", "error_with_states", "error_with_extra_components"
    "function": "error_with_vel",
    "average_length": 1,  # use average of last 5 sensor data points
}

# for more information look at the docstring of DirectControl.create_reward()
reward_options = {
    "function": "normal",  # add your own reward function if you want to
    "discrete_bonus": True,
    "oscillation_pen_dependent_on_error": True,
    "oscillation_pen_fun": np.sqrt,
    "oscillation_pen_gain": 0.1,
    "error_pen_fun": None,
}

env_options = {
    "online_sys": DirectControllerOnlineConnection(),
    "sensor_freq": 4_000,  # generate sensor data with 4_000 Hz
    "output_freq": 100,  # update output with 100 Hz
    "observation_kwargs": observation_options,
    "reward_kwargs": reward_options,
    "log": False,  # log false for training envs
}

policy_options = {
    "actor_layers": 0,
    "actor_layer_width": 10,  # amount of neurons per layer in the hidden layers
    "actor_activation_fun": th.nn.Tanh(),
    "actor_end_activation_fun": th.nn.Tanh(),  # must be a activation function that clips the value between (-1, 1)
    "actor_bias": False,
    "critic_layers": 2,
    "critic_layer_width": 200,  # amount of neurons per layer in the hidden layers
    "critic_activation_fun": th.nn.Tanh(),
    "critic_bias": True,
}

rl_options = {
    "save_path": "controller_test_online",
    "tensorboard_log_name": "tensorboard_controller_test_online",
    "timesteps": 100_000,
    "action_noise": 0.05,
}

params_dict = {"env_options": env_options,
               "policy_options": policy_options,
               "rl_options": rl_options}


if __name__ == "__main__":
    # check for existing entries and add a new one
    pathlib.Path(rl_options["save_path"]).mkdir(exist_ok=True)
    dir_entries = os.listdir(rl_options["save_path"])
    if dir_entries:
        dir_entries_int = [int(d) for d in dir_entries]
        run_nbr = max(dir_entries_int) + 1
    else:
        run_nbr = 0

    # create multiple envs for parallel use
    env = DirectControllerOnline(**env_options)

    env_options["log"] = True  # log only true for evaluation envs

    # create normal action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(rl_options["action_noise"]) * np.ones(n_actions))

    # create callback that logs matplotlib figures on tensorboard and saves possible best model
    online_eval_env = DirectControllerOnline(**env_options)  # create eval env
    online_eval_env = Monitor(online_eval_env)
    eval_callback = CustomEvalCallback(online_eval_env,
                                       eval_freq=1500,
                                       deterministic=True,
                                       best_model_save_path=f"{rl_options['save_path']}\\{run_nbr}\\best_model")

    callbacks = CallbackList([eval_callback])

    # # use DDPG and create a tensorboard
    # # start tensorboard server with tensorboard --logdir ./{tensorboard_log}/
    model = DDPG(custom_policy.CustomDDPGPolicy,
                 env,
                 learning_starts=6000,  # delay start to compensate for bad starting conditions
                 verbose=2,
                 action_noise=action_noise,
                 tensorboard_log=rl_options["tensorboard_log_name"],
                 policy_kwargs=policy_options,
                 )

    # model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\controller_test_online\5\model.zip", env, force_reset=True,
    #                   custom_objects={"learning_starts": 0,
    #                                   "action_noise": action_noise})

    model.learn(total_timesteps=rl_options["timesteps"], tb_log_name=f"{run_nbr}", callback=callbacks)
    # #
    # # # save model if you want to
    model.save(f"{rl_options['save_path']}\\{run_nbr}\\model")

    with open(f"{rl_options['save_path']}\\{run_nbr}\\extra_info.json", 'w+') as f:
        json.dump(params_dict, f, indent=4, default=utils.custom_default_json)

    utils.export_onnx(model, f"{rl_options['save_path']}\\{run_nbr}\\model.onnx")


