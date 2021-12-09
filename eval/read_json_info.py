import json
import os
from stable_baselines3 import DDPG


main_folder = "eval_discrete_reward_acotr_size_critic_size_af"
rmse = []
for dr in os.listdir(f"{main_folder}"):
    with open(f"C:\AktiveProjekte\Python\Projektmodul\eval\{main_folder}\{dr}\extra_info.json", "r") as f:
        info_dict = json.load(f)
        rmse.append((dr, info_dict["rmse"]))


rmse.sort(key=lambda x: x[1])
for entry in rmse:
    print(entry)