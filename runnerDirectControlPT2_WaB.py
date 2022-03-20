import multiprocessing

import numpy as np
import wandb
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from CustomEvalCallback import CustomEvalCallback
from envs.DirectControllerSim import DirectControllerSim
import custom_policy_no_bias
from torch import nn

if __name__ == "__main__":
    hyperparameter_defaults = {
        "actor_layers": 2,
        "actor_activation_fun": "nn.Tanh()",
        "critic_layers": 2,
        "critic_layer_width": 50,
        "critic_activation_fun": "nn.ReLU()",
        "obs_i": True,
        "obs_d": True,
        "obs_input_vel": True,
        "obs_output_vel": False,
        "obs_output": False,
        "obs_input": False,
        "reward_fun": "discrete_u_pen_dep_on_error",
    }

    run = wandb.init(
        project="sb3",
        config=hyperparameter_defaults,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    config = wandb.config
    print(config)

    obs_config = {
        "i": config["obs_i"],
        "d": config["obs_d"],
        "input_vel": config["obs_input_vel"],
        "output_vel": config["obs_output_vel"],
        "output": config["obs_output"],
        "input": config["obs_input"]
    }
    print(multiprocessing.cpu_count() - 1)
    env = make_vec_env(DirectControllerSim, multiprocessing.cpu_count() - 1, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"obs_config": obs_config,
                                   "reward_function": config["reward_fun"]})  # create learning env
    # env = DirectControllerSim()

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    # create eval callback
    online_eval_env = DirectControllerSim(log=True, obs_config=obs_config, reward_function=config["reward_fun"])  # create eval env
    online_eval_env = Monitor(online_eval_env)
    eval_callback = CustomEvalCallback(online_eval_env, eval_freq=3000, deterministic=True,
                                       best_model_save_path=f"models/{run.id}\\best_model")
    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
    # create callback list
    callbacks = CallbackList([eval_callback, wandb_callback])

    # # use DDPG and create a tensorboard
    # # start tensorboard server with tensorboard --logdir ./{tensorboard_log}/
    policy_kwargs = {"actor_layers": config["actor_layers"],
                     "actor_activation_fun": eval(config["actor_activation_fun"]),
                     "critic_layers": config["critic_layers"],
                     "critic_layer_width": config["critic_layer_width"],
                     "critic_activation_fun": eval(config["critic_activation_fun"]),
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
    eval_info = DirectControllerSim(log=True,
                                    obs_config=obs_config,
                                    reward_function=config["reward_fun"]
                                    ).eval(model, folder_name=f"{run.dir}\custom_eval")

    wandb.log(eval_info)
    wandb.finish()
