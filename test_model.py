from stable_baselines3 import DDPG
from envs.DirectControllerSim import DirectControllerSim
import matplotlib.pyplot as plt
import numpy as np

env = DirectControllerSim()

model = DDPG.load(r"C:\AktiveProjekte\Python\Projektmodul2\eval\RUN\22\model.zip", env)

obs = env.reset()
error = np.linspace(-1e-2, 1e-2, 100)
actions = []
print(obs)
for e in error:
    obs[2] = e
    action, _ = model.predict(obs)
    actions.append(action)

plt.plot(error, actions)
plt.show()

# while True:
#     done = False
#     obs = env.reset()
#     print(obs)
#     while not done:
#         # obs[4] = 0
#         action, _ = model.predict(obs)
#         # action = [0]
#
#         obs, reward, done, info = env.step(action)
#         # done = False
#
#     plt.plot(env.w)
#     plt.plot(env.sim._sim_out)
#     plt.grid()
#     plt.show()