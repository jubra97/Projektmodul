import json
import os

dirs = os.listdir("statistical_scattering4")

extra_infos = []
for dir in dirs:
    name = f"statistical_scattering4\\{dir}"
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

extra_infos = sorted(extra_infos, key=lambda x: x["rmse"])
for info in extra_infos:
    print(info["name"])
    print(info["rmse"])