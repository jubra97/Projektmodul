import gym
import numpy as np
import json
import matplotlib
from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.a2c.policies import ActorCriticPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, TD3, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from envs.DirectControllerPT2 import DirectControllerPT2
from tensorboard_logger import TensorboardCallback
import matplotlib.pyplot as plt
from CustomEvalCallback import CustomEvalCallback
from stable_baselines3.common.monitor import Monitor


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
env = DirectControllerPT2()
env = Monitor(env)

# create action noise
n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.005) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

# create tensorboard callback
tb_callback = TensorboardCallback(env)

# create eval callback

eval_callback = CustomEvalCallback(DirectControllerPT2, eval_freq=1500, deterministic=True)

# create callback list
callbacks = CallbackList([tb_callback, eval_callback])

# # use DDPG and create a tensorboard
# # start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
policy_kwargs = dict(net_arch=[16, 16])
model = TD3(MlpPolicy, env, learning_starts=1500, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/", policy_kwargs=policy_kwargs, batch_size=300)
# model = TD3("MlpPolicy", env, verbose=0, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/", batch_size=300, gamma=0.1)#, policy_kwargs=policy_kwargs)
# model = A2C(ActorCriticPolicy, env, verbose=0, tensorboard_log="./dmsSim_ddpg_tensorboard/")  # , policy_kwargs=policy_kwargs)
model.learn(total_timesteps=2000000, tb_log_name="first_run", callback=callbacks)

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
with open("log_data_last_run.json2", "w") as f:
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
    matplotlib.use("TkAgg")
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
