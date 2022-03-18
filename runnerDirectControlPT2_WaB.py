import numpy as np
import torch as th
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from torch import nn
from CustomEvalCallback import CustomEvalCallback
from envs.DirectControllerSim import DirectControllerSim
import custom_policy_no_bias
import multiprocessing

if __name__ == "__main__":

    hyperparameter_defaults = {
        "actor_layers": 2,
        "actor_layer_width": 5,
        "actor_activation_fun": "nn.Tanh()",
        "actor_end_activation_fun": "nn.Hardtanh()",
        "critic_layers": 1,
        "critic_layer_width": 50,
        "critic_activation_fun": "nn.ReLU()",
        "critic_bias": True
    }

    run = wandb.init(
        project="sb3",
        config=hyperparameter_defaults,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    config = wandb.config
    print(config)


    print(multiprocessing.cpu_count()-1)
    env = make_vec_env(DirectControllerSim, multiprocessing.cpu_count()-1, vec_env_cls=SubprocVecEnv)  # create learning env
    # env = DirectControllerSim()

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    # create eval callback
    online_eval_env = DirectControllerSim(log=True)  # create eval env
    online_eval_env = Monitor(online_eval_env)
    eval_callback = CustomEvalCallback(online_eval_env, eval_freq=3000, deterministic=True,
                                       best_model_save_path=f"models/{run.id}\\best_model")
    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
    # create callback list
    callbacks = CallbackList([eval_callback, wandb_callback])

    # # use DDPG and create a tensorboard
    # # start tensorboard server with tensorboard --logdir ./{tensorboard_log}/
    policy_kwargs = {"actor_layers": config["actor_layers"],
                     "actor_layer_width": config["actor_layer_width"],
                     "actor_activation_fun": eval(config["actor_activation_fun"]),
                     "actor_end_activation_fun": eval(config["actor_end_activation_fun"]),
                     "critic_layers": config["critic_layers"],
                     "critic_layer_width": config["critic_layer_width"],
                     "critic_activation_fun": eval(config["critic_activation_fun"]),
                     "critic_bias": config["critic_bias"],
                     }

    model = DDPG("CustomTD3Policy",
                 env,
                 learning_starts=3000,
                 verbose=2,
                 action_noise=action_noise,
                 tensorboard_log=f"{run.dir}\\tensorboard",
                 policy_kwargs=policy_kwargs,
                 train_freq=1,
                 gradient_steps=1,
                 learning_rate=1e-3
                 )
    model.learn(total_timesteps=150_000, tb_log_name=f"{run.id}", callback=callbacks)
    eval_info = DirectControllerSim(log=True).eval(model, folder_name=f"{run.dir}\custom_eval")

    wandb.log(eval_info)
    wandb.finish()
