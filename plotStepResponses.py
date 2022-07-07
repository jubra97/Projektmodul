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

        rise_start = 0.1 * 0.5
        rise_stop = 0.9 * 0.5
        start_time = int(env.sim.model_freq * 0.5)
        index_start = np.argmax(np.array(env.sim._sim_out)[start_time:] > rise_start)
        index_end = np.argmax(np.array(env.sim._sim_out)[start_time + index_start:] > rise_stop)
        rise_time = index_end / env.sim.model_freq
        if rise_time == 0:
            rise_time = 1
        print(rise_time)

        # calculate setting time with 5% band
        lower_bound = 0.5 - 0.5 * 0.05
        upper_bound = 0.5 + 0.5 * 0.05
        # go backwards through sim._sim_out and find first index out of bounds
        index_lower_out = list(np.array(env.sim._sim_out)[::-1] < lower_bound)
        index_lower = 0
        try:
            index_lower = index_lower_out.index(True)
        except ValueError:
            index_lower = env.sim.n_sample_points

        index_upper_out = list(np.array(env.sim._sim_out)[::-1] > upper_bound)
        index_upper = 0
        try:
            index_upper = index_upper_out.index(True)
        except ValueError:
            index_upper = env.sim.n_sample_points
        # index_lower = list(np.array(env.sim._sim_out)[::-1] < lower_bound).index(True)
        # index_lower = index_lower if index_lower > 0 else env.sim.n_sample_points
        # index_upper = list(np.array(env.sim._sim_out)[::-1] > upper_bound).index(True)
        # index_upper = index_upper if index_upper > 0 else env.sim.n_sample_points
        last_out_of_bounds = min([index_lower, index_upper])

        setting_time = (env.sim.n_sample_points - last_out_of_bounds - start_time) / env.sim.model_freq

        # if last_out_of_bounds >= (env.sim.n_sample_points - start_time):
        #     setting_time = 1
        # else:
        #     setting_time = env.sim.n_sample_points - last_out_of_bounds - start_time) / env.sim.model_freq
        print(setting_time)
        print(np.mean(np.sqrt(np.square(np.array(env.w) - np.array(env.sim._sim_out)))))


# env = DirectControllerSim(**env_options)
# for _ in range(1):
#     time.sleep(0.5)
#     done = False
#     obs = env.reset(step_start=0, step_end=0.5, step_slope=0)
#     actions = [0] * 50 + [0.25] + [0] * 99
#     i = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step([actions[i]])
#         i += 1
# envs.append(env)

import matplotlib.pyplot as plt

font = {'size': 14}

plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (9, 5)

i = 0
plt.plot(env.sim.t, env.w, label="w")
for env in envs:
    i += 1
    plt.plot(env.sim.t, env.sim._sim_out, label=f"y; run: {i}")

plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Normalized Force [N]")
plt.xlim([0, 1.5])
plt.legend()
plt.tight_layout()
plt.show()