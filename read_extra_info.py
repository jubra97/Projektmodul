import json
import os

dirs = os.listdir("statistical_scattering3")

extra_infos = []
for dir in dirs:
    name = f"statistical_scattering3\\{dir}"
    if int(dir) < 10:
        with open(name + "\\extra_info.json", "r") as f:
            extra_infos.append(json.load(f))
            print(f"Name {dir}")
            print(f"RMSE: {extra_infos[-1]['rmse']}")
            print(f"Setting Time: {extra_infos[-1]['mean_setting_time']}")
            print(f"Riste Time: {extra_infos[-1]['mean_rise_time']}")
