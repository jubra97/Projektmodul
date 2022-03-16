from stable_baselines3 import DDPG
from envs.DirectControllerSim import DirectControllerSim
import matplotlib.pyplot as plt
import numpy as np
import torch as th

env = DirectControllerSim()
env2 = DirectControllerSim()

# model = DDPG.load(r"C:\AktiveProjekte\Python\Projektmodul2\eval\RUN\22\model.zip", env)
model = DDPG.load(r"G:\Projektmodul_Julius_Valentin\eval\RUN\32\model.zip", env)

print(model)
# obs = env.reset()
# error = np.linspace(-1e-2, 1e-2, 100)
# actions = []
# print(obs)
# for e in error:
#     obs[2] = e
#     action, _ = model.predict(obs)
#     actions.append(action)
#
# plt.plot(error, actions)
# plt.show()

while True:
    done = False
    obs = env.reset(0.1, 0.3, 0.5)
    env2.reset(0.1, 0.3, 0.5)
    print(obs)

    actions = []
    actions2 = []
    while not done:
        # obs[4] = 0
        action, _ = model.predict(obs)
        print(f"Predicted Action: {action}")
        own_action = th.matmul(model.actor.mu._modules["0"].weight.data, th.tensor(obs, dtype=th.float32))
        print(f"Own Predicted Action: {own_action}")
        actions.append(action)
        actions2.append(own_action)
        # action = [0]

        obs, reward, done, info = env.step(action)
        obs2, reward2, done2, info2 = env2.step(own_action)
        # done = False



    fig, ax = plt.subplots(1, 2)
    ax[0].plot(env.w)
    ax[0].plot(env.sim._sim_out)
    ax[0].plot(env2.sim._sim_out)
    ax[0].plot(env.sim._u)
    ax[0].plot(env2.sim._u)
    ax[0].grid()

    ax[1].plot(np.array(actions) * 100)
    ax[1].plot(np.array(actions2) * 100)
    ax[1].grid()

    plt.show()
