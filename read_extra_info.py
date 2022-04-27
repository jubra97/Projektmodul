import json
import os

dirs = os.listdir("controller_test")

extra_infos = []
for dir in dirs:
    if int(dir) >= 106 and int(dir) < 113:
        name = f"controller_test\\{dir}"
        with open(name + "\\extra_info.json", "r") as f:
            extra_infos.append(json.load(f))
            print(dir)
            print(extra_infos[-1]["rmse"])
            print(extra_infos[-1]["mean_setting_time"])
            print(extra_infos[-1]["mean_rise_time"] - 0.5)
