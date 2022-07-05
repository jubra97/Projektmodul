import os
import pathlib
import wandb
import shutil

import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import custom_policy
from CustomEvalCallback import CustomEvalCallback
from ActionNoiseCallback import ActionNoiseCallback
from envs.DirectControllerSim import DirectControllerSim


if __name__ == "__main__":
    hyperparameter_defaults = {
    }

    run = wandb.init(
        project="Projektmodul_Ergebnisse",
        config=hyperparameter_defaults,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    observation_options = {
        "function": "error_with_vel",
    }

    # for more information look at the docstring of DirectControl.create_reward()
    reward_options = {
        "function": "normal",  # add your own reward function if you want to
        "discrete_bonus": True,
        "oscillation_pen_dependent_on_error": False,
        "oscillation_pen_fun": np.sqrt,
        "oscillation_pen_gain": 25,
        "error_pen_fun": None,
    }

    env_options = {
        "model_freq": 12_000,  # sim with 12_000 Hz
        "sensor_freq": 4_000,  # generate sensor data with 4_000 Hz
        "output_freq": 100,  # update output with 100 Hz
        "observation_kwargs": observation_options,
        "reward_kwargs": reward_options,
        "log": False,  # log false for training envs
    }

    policy_options = {
        "actor_layers": 2,
        "actor_layer_width": 100,  # amount of neurons per layer in the hidden layers
        "actor_activation_fun": th.nn.ReLU(),
        "actor_end_activation_fun": th.nn.Tanh(),  # must be a activation function that clips the value between (-1, 1)
        "actor_bias": False,
        "critic_layers": 2,
        "critic_layer_width": 200,  # amount of neurons per layer in the hidden layers
        "critic_activation_fun": th.nn.ReLU(),
        "critic_bias": True,
    }

    rl_options = {
        "save_path": "wandb_base_config",
        "tensorboard_log_name": "tensorboard_base_config",
        "cpu_cores": 3,
        "timesteps": 300_000,
        "action_noise": (0.1, 0.0003, 250_000)
    }

    params_dict = {"env_options": env_options,
                   "policy_options": policy_options,
                   "rl_options": rl_options}

    # create multiple envs for parallel use
    env = make_vec_env(DirectControllerSim, rl_options["cpu_cores"], vec_env_cls=SubprocVecEnv, env_kwargs=env_options)
    # env= DirectControllerSim(**env_options)

    env_options["log"] = True  # log only true for evaluation envs

    # create callback that logs matplotlib figures on tensorboard and saves possible best model
    online_eval_env = DirectControllerSim(**env_options)  # create eval env
    online_eval_env = Monitor(online_eval_env)
    eval_callback = CustomEvalCallback(online_eval_env,
                                       eval_freq=1500,
                                       deterministic=True,
                                       best_model_save_path=f"{run.dir}\\best_model")
    wandb_callback = WandbCallback(model_save_path=f"{run.dir}", verbose=2)
    callbacks = CallbackList([eval_callback, wandb_callback])

    if isinstance(rl_options["action_noise"], tuple) and len(rl_options["action_noise"]):
        action_noise_callback = ActionNoiseCallback(rl_options["action_noise"][0], rl_options["action_noise"][1],
                                                    rl_options["action_noise"][2])
        callbacks.callbacks.append(action_noise_callback)
        action_noise = None
    elif rl_options["action_noise"]:
        # create normal action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=float(rl_options["action_noise"]) * np.ones(n_actions))
    else:
        action_noise = None

    # # use DDPG and create a tensorboard
    # # start tensorboard server with tensorboard --logdir ./{tensorboard_log}/
    model = DDPG(custom_policy.CustomDDPGPolicy,
                 env,
                 learning_starts=3000,  # delay start to compensate for bad starting conditions
                 verbose=2,
                 action_noise=action_noise,
                 tensorboard_log=f"{run.dir}",
                 policy_kwargs=policy_options,
                 train_freq=1,
                 gradient_steps=1,
                 learning_rate=1e-3,
                 )
    model.learn(total_timesteps=rl_options["timesteps"], tb_log_name=f"{run.id}", callback=callbacks)
    eval_info = DirectControllerSim(**env_options).eval(model,
                                            folder_name=f"{run.dir}/custom_eval",
                                            options_dict=params_dict)
    # #
    # # # save model if you want to
    model.save(f"{run.dir}\\model")

    shutil.copytree(f"{run.dir}/custom_eval", f"{rl_options['save_path']}/{run.id}")
    shutil.copy2(f"{run.dir}/best_model.zip", f"{rl_options['save_path']}/{run.id}")
    shutil.copy2(f"{run.dir}/model.zip", f"{rl_options['save_path']}/{run.id}")


    wandb.log(eval_info)
    wandb.finish()