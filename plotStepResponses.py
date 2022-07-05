import json
import time
import numpy as np
import torch as th
import os

from envs.DirectControllerSim import DirectControllerSim
from stable_baselines3 import DDPG

start_path = r"C:\AktiveProjekte\Python\Projektmodul2\wandb_base_config"

if __name__ == "__main__":
    envs = []
    for run in os.listdir(start_path):
        run_path = start_path + '\\' + run
        with open(f"{run_path}\\extra_info.json", 'r') as f:
            params_dict = json.load(f)

        observation_options = params_dict["env_options"]["observation_kwargs"]
        reward_options = params_dict["env_options"]["reward_kwargs"]
        for key, value in reward_options.items():
            if "fun" in key[-3:]:
                if value:
                    reward_options[key] = eval(f"np.{value}")
        policy_options = params_dict["policy_options"]
        for key, value in policy_options.items():
            if "fun" in key[-3:]:
                if value:
                    policy_options[key] = eval(f"th.nn.{value}")
        env_options = params_dict["env_options"]

        env = DirectControllerSim(**env_options)
        model = DDPG.load(f"{run_path}\\model.zip", env)

        # info = env.eval(model, r"tmp" + "\\" + run, {})
        # if info["rmse"] < 0.05:
        #     print("_____________________________________________________________________________________")
        #     print(run)
        #     print(observation_options)
        #     print("_____________________________________________________________________________________")
    # obs = env.reset()
    #
        for _ in range(1):
            time.sleep(0.5)
            done = False
            obs = env.reset(step_start=0, step_end=0.5, step_slope=0)
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
        envs.append(env)

    env = DirectControllerSim(**env_options)
    for _ in range(1):
        time.sleep(0.5)
        done = False
        obs = env.reset(step_start=0, step_end=0.5, step_slope=0)
        actions = [0] * 50 + [0.25] + [0] * 99
        i = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step([actions[i]])
            i += 1
    envs.append(env)

    import matplotlib.pyplot as plt
    for env in envs:
        plt.plot(env.sim.t_sensor, env.sim.sensor_out)
    # plt.plot(env.sim.t_sensor, env.w)
    plt.grid()
    plt.show()