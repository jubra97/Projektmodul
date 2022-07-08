import json
import time
import numpy as np
import torch as th
import os

from envs.DirectControllerSim import DirectControllerSim
from stable_baselines3 import DDPG

start_path = r"C:\AktiveProjekte\Python\Projektmodul2\wandb_observations"
names = []

if __name__ == "__main__":
    envs = []
    for run in os.listdir(start_path):
        run_path = start_path + '\\' + run
        if os.path.isfile(run_path):
            continue
        with open(f"{run_path}\\custom_eval\\extra_info.json", 'r') as f:
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

        if params_dict["mean_rise_time"] > 0.5 or params_dict["mean_setting_time"] > 0.5:
            continue

        obs_fun = params_dict["env_options"]["observation_kwargs"].get("function")
        obs_config = list(params_dict["env_options"]["observation_kwargs"].get("obs_config", {}).keys())
        obs_hist = params_dict["env_options"]["observation_kwargs"].get("history_length", None)
        obs_bias = params_dict["policy_options"].get("actor_bias")

        dict_key = obs_fun
        if obs_config:
            dict_key += f", {obs_config}"
        if obs_hist:
            dict_key += f", {obs_hist}"

        # print(dict_key)
        if dict_key == "error_with_vel":
            dict_key = "$e, \dot{e}, \dot{u}$"
        if dict_key == "error_with_extra_components":
            continue
            dict_key = "$e$"
        if dict_key == "error_with_extra_components, ['d', 'i']":
            dict_key = "$e, \dot{e}, \int{e}$"
        if dict_key == "error_with_extra_components, ['d']":
            continue
            dict_key = "$e, \dot{e}$"
        if dict_key == "error_with_extra_components, ['d', 'input']":
            continue
            dict_key = "$e, \dot{e}, u$"
        if dict_key == "error_with_extra_components, ['d', 'output']":
            continue
            dict_key = "$e, \dot{e}, y$"
        if dict_key == "error_with_extra_components, ['output']":
            continue
            dict_key = "$e, y$"
        if dict_key == "error_with_extra_components, ['input']":
            continue
            dict_key = "$e, u$"
        if dict_key == "error_with_last_states, 1":
            continue
            dict_key = "$e$"
        if dict_key == "error_with_last_states, 2":
            continue
            dict_key = "$e, e_{t-1}$"
        if dict_key == "error_with_last_states, 3":
            continue
            dict_key = "$e, e_{t-1}, e_{t-2}$"
        if dict_key == "raw_with_last_states, 1":
            continue
            dict_key = "$w, u, y$"
        if dict_key == "raw_with_last_states, 2":
            continue
            dict_key = "$w, w_{t-1}$ \n $u, u_{t-1}$ \n $y, y_{t-1} $"
        if dict_key == "raw_with_last_states, 3":
            if obs_bias:
                continue
            dict_key = "$w, w_{t-1}, w_{t-2}$ \n $u, u_{t-1}, u_{t-2}$ \n $y, y_{t-1}, y_{t-2}$"
        if dict_key == "raw_with_vel":
            continue
            dict_key = "$w, \dot{w}, u, \dot{u}, y, \dot{y}$"

        names.append(dict_key)


        # info = env.eval(model, r"tmp" + "\\" + run, {})
        # if info["rmse"] < 0.05:
        #     print("_____________________________________________________________________________________")
        #     print(run)
        #     print(observation_options)
        #     print("_____________________________________________________________________________________")
    # obs = env.reset()
    #

        done = False
        obs = env.reset(step_start=0.0, step_end=0.5, step_slope=0.2)
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
        # names.append(run)


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

tmp_dict = {}

font = {'size': 14}

plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (9, 5)

i = 0
print(names)
env.reset(step_start=0.0, step_end=0.5, step_slope=0.2)
plt.plot(env.sim.t, env.w, label="w")
tmp_dict["w"] = list(env.w)
tmp_dict["t"] = list(env.sim.t)
tmp_dict["y"] = {}
for env in envs:
    tmp_dict["y"][names[i]] = list(env.sim._sim_out)
    plt.plot(env.sim.t, env.sim._sim_out, label=f"{names[i]}")
    i += 1


with open("tmp_step_response_01_error.json", "w") as f:
    json.dump(tmp_dict, f, indent=4)

plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Normalized Force [N]")
plt.xlim([0, 1.5])
plt.legend()
plt.tight_layout()
plt.show()