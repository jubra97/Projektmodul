import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn

main_path = r"C:\AktiveProjekte\Python\Projektmodul2\wandb_observations"
dirs = os.listdir(main_path)
extra_infos = []
for dir in dirs:
    name = f"{main_path}\\{dir}"
    try:
        with open(name + "\\custom_eval\\extra_info.json", "r") as f:
            tmp = json.load(f)
            tmp["name"] = dir
            extra_infos.append(tmp)
            # print(f"Name {dir}")
            # print(f"RMSE: {extra_infos[-1]['rmse']}")
            # print(f"Setting Time: {extra_infos[-1]['mean_setting_time']}")
            # print(f"Riste Time: {extra_infos[-1]['mean_rise_time']}")
    except FileNotFoundError:
        ...


out_dict = {}
out_dict2 = {}
dict_keys = []
extra_infos = sorted(extra_infos, key=lambda x: x["rmse"])
for info in extra_infos:
    # print(info["name"])
    # print(info["rmse"])
    # print(info["policy_options"]["actor_bias"])

    obs_fun = info["env_options"]["observation_kwargs"].get("function")
    obs_config = list(info["env_options"]["observation_kwargs"].get("obs_config", {}).keys())
    obs_hist = info["env_options"]["observation_kwargs"].get("history_length", None)
    obs_bias = info["policy_options"].get("actor_bias")


    dict_key = obs_fun
    if obs_config:
        dict_key += f", {obs_config}"
    if obs_hist:
        dict_key += f", {obs_hist}"



    # print(dict_key)
    if dict_key == "error_with_vel":
        dict_key = "$e, \dot{e}, \dot{u}$"
    if dict_key == "error_with_extra_components":
        dict_key = "$e$"
    if dict_key == "error_with_extra_components, ['d', 'i']":
        dict_key = "$e, \dot{e}, \int{e}$"
    if dict_key == "error_with_extra_components, ['d']":
        dict_key = "$e, \dot{e}$"
    if dict_key == "error_with_extra_components, ['d', 'input']":
        dict_key = "$e, \dot{e}, u$"
    if dict_key == "error_with_extra_components, ['d', 'output']":
        dict_key = "$e, \dot{e}, y$"
    if dict_key == "error_with_extra_components, ['output']":
        dict_key = "$e, y$"
    if dict_key == "error_with_extra_components, ['input']":
        dict_key = "$e, u$"
    if dict_key == "error_with_last_states, 1":
        dict_key = "$e$"
    if dict_key == "error_with_last_states, 2":
        dict_key = "$e, e_{t-1}$"
    if dict_key == "error_with_last_states, 3":
        dict_key = "$e, e_{t-1}, e_{t-2}$"
    if dict_key == "raw_with_last_states, 1":
        dict_key = "$w, u, y$"
    if dict_key == "raw_with_last_states, 2":
        dict_key = "$w, w_{t-1}$ \n $u, u_{t-1}$ \n $y, y_{t-1} $"
    if dict_key == "raw_with_last_states, 3":
        dict_key = "$w, w_{t-1}, w_{t-2}$ \n $u, u_{t-1}, u_{t-2}$ \n $y, y_{t-1}, y_{t-2}$"
    if dict_key == "raw_with_vel":
        dict_key = "$w, \dot{w}, u, \dot{u}, y, \dot{y}$"

    print(info["name"])
    print(dict_key)
    print(obs_bias)
    print(info["rmse"])

    dict_keys.append(dict_key)

dict_keys = sorted(dict_keys)
for dict_key in dict_keys:
    out_dict[dict_key] = {}
    out_dict2[dict_key] = {}


for info in extra_infos:

    obs_fun = info["env_options"]["observation_kwargs"].get("function")
    obs_config = list(info["env_options"]["observation_kwargs"].get("obs_config", {}).keys())
    obs_hist = info["env_options"]["observation_kwargs"].get("history_length", None)
    obs_bias = info["policy_options"].get("actor_bias")

    dict_key = obs_fun
    if obs_config:
        dict_key += f", {obs_config}"
    if obs_hist:
        dict_key += f", {obs_hist}"
    if dict_key == "error_with_vel":
        dict_key = "$e, \dot{e}, \dot{u}$"
    if dict_key == "error_with_extra_components":
        dict_key = "$e$"
    if dict_key == "error_with_extra_components, ['d', 'i']":
        dict_key = "$e, \dot{e}, \int{e}$"
    if dict_key == "error_with_extra_components, ['d']":
        dict_key = "$e, \dot{e}$"
    if dict_key == "error_with_extra_components, ['d', 'input']":
        dict_key = "$e, \dot{e}, u$"
    if dict_key == "error_with_extra_components, ['d', 'output']":
        dict_key = "$e, \dot{e}, y$"
    if dict_key == "error_with_extra_components, ['output']":
        dict_key = "$e, y$"
    if dict_key == "error_with_extra_components, ['input']":
        dict_key = "$e, u$"
    if dict_key == "error_with_last_states, 1":
        dict_key = "$e$"
    if dict_key == "error_with_last_states, 2":
        dict_key = "$e, e_{t-1}$"
    if dict_key == "error_with_last_states, 3":
        dict_key = "$e, e_{t-1}, e_{t-2}$"
    if dict_key == "raw_with_last_states, 1":
        dict_key = "$w, u, y$"
    if dict_key == "raw_with_last_states, 2":
        dict_key = "$w, w_{t-1}$ \n $u, u_{t-1}$ \n $y, y_{t-1} $"
    if dict_key == "raw_with_last_states, 3":
        dict_key = "$w, w_{t-1}, w_{t-2}$ \n $u, u_{t-1}, u_{t-2}$ \n $y, y_{t-1}, y_{t-2}$"
    if dict_key == "raw_with_vel":
        dict_key = "$w, \dot{w}, u, \dot{u}, y, \dot{y}$"

    out_dict[dict_key][obs_bias] = info["mean_rise_time"] + info["mean_setting_time"]
    out_dict2[dict_key][obs_bias] = f"{info['mean_rise_time']:.2f}\n{info['mean_setting_time']:.2f}"
#
# print(out_dict)
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
df = pd.DataFrame.from_dict(out_dict)
df2 = pd.DataFrame.from_dict(out_dict2)
print(df)
plt.figure(figsize=(5, 7))
h_map = seaborn.heatmap(df.T, annot=df2.T, fmt="s", linewidths=1.5, cmap="Greens_r", cbar=False, annot_kws={"fontsize": 16})
# h_map.set(xlabel="Actor Bias", ylabel="Given Observation", fontsize=14)
h_map.set_xlabel("Actor Bias", fontsize=22)
h_map.set_ylabel("Given Observation", fontsize=22)
# h_map.set_title("RMSE nach gegebener Observation und Actor Bias", fontsize=16)
h_map.set_yticklabels(labels=h_map.get_yticklabels(), ha="right", linespacing=0.7, fontsize=20)
h_map.set_xticklabels(labels=h_map.get_xticklabels(), linespacing=0.7, fontsize=20)
# h_map.yaxis.major_ticklabels.set_ha("center")
# h_map.yaxis.set_tick_params(pad=51)
# plt.yticks(np.arange(12)+0.5, va="center", ha="center")
plt.tight_layout()
plt.show()
