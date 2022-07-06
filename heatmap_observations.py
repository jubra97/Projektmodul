import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import json
import os
import pandas as pd


main_path = r"G:\Projektmodul_Julius_Valentin\wandb_observations"
dirs = os.listdir(main_path)
extra_infos = []
for dir in dirs:
    name = f"{main_path}\\{dir}"
    try:
        with open(name + "\\extra_info.json", "r") as f:
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
dict_keys = []
extra_infos = sorted(extra_infos, key=lambda x: x["rmse"])
for info in extra_infos:
    print(info["name"])
    print(info["rmse"])
    print(info["policy_options"]["actor_bias"])

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
    dict_keys.append(dict_key)

dict_keys = sorted(dict_keys)
for dict_key in dict_keys:
    out_dict[dict_key] = {}


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

    out_dict[dict_key][obs_bias] = info["rmse"]

# print(out_dict)

df = pd.DataFrame.from_dict(out_dict)
print(df)

seaborn.heatmap(df.T, annot=True, linewidths=.5, cmap="Greens_r")
plt.show()
