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

from CustomEvalCallback import CustomEvalCallback
from envs.DirectControllerSim import DirectControllerSim
import custom_policy_no_bias

actor_net = [5, 5]
critic_net = [200, 200]
af = th.nn.Tanh


if __name__ == "__main__":

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 25000,
        "env_name": DirectControllerSim(),
    }




    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # model = DDPG(config["policy_type"], config["env_name"], verbose=1, tensorboard_log=f"runs/{run.id}")

    dir = os.listdir(r"eval\RUN")
    if dir:
        dir_int = [int(d) for d in dir]
        max_run_nbr = max(dir_int)
    else:
        max_run_nbr = 0
    run_nbr = max_run_nbr + 1

    RUN_NAME = f"RUN\\{run_nbr}"

    env = make_vec_env(DirectControllerSim, 3, vec_env_cls=SubprocVecEnv)  # create learning env
    # env = DirectControllerSim()

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    # create eval callback
    online_eval_env = DirectControllerSim(log=True)  # create eval env
    online_eval_env = Monitor(online_eval_env)
    eval_callback = CustomEvalCallback(online_eval_env, eval_freq=1500, deterministic=True, best_model_save_path=f"eval\\{RUN_NAME}\\best_model")
    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
    # create callback list
    callbacks = CallbackList([eval_callback, wandb_callback])

    # # use DDPG and create a tensorboard
    # # start tensorboard server with tensorboard --logdir ./{tensorboard_log}/
    policy_kwargs = dict(net_arch=dict(pi=actor_net, qf=critic_net), activation_fn=af)

    model = DDPG("CustomTD3Policy",
                 env,
                 learning_starts=3000,
                 verbose=2,
                 action_noise=action_noise,
                 tensorboard_log="./ddpg_direct_control2/",
                 policy_kwargs=policy_kwargs,
                 train_freq=1,
                 gradient_steps=1,
                 learning_rate=1e-3
                 )
    model.learn(total_timesteps=10_000, tb_log_name=f"{RUN_NAME}", callback=callbacks)
    DirectControllerSim(log=True).eval(model, folder_name=RUN_NAME)
    # #
    # # # save model if you want to
    model.save(f"eval\\{RUN_NAME}\\model")

    wandb.finish()