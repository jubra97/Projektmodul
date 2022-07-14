import json
import time
import numpy as np
import torch as th

import utils
from envs.DirectControllerOnline import DirectControllerOnline
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from stable_baselines3 import DDPG

run_path = r"C:\Users\brandlju\PycharmProjects\Projektmodul\controller_test_online2\0"

if __name__ == "__main__":
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
    env_options["online_sys"] = DirectControllerOnlineConnection()

    env = DirectControllerOnline(**env_options)
    model = DDPG.load(f"{run_path}\\model.zip", env)

    obs = env.reset()
    # while True:
    #     action, _ = model.predict(obs)
    #     obs, reward, done, info = env.step()

    obss = []
    for _ in range(10):
        time.sleep(0.5)
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            done = False

    #         obss.append(obs[-2])
    #
    # import matplotlib.pyplot as plt
    # plt.plot(obss)
    # plt.grid()
    # plt.show()


