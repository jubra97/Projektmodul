from stable_baselines3.ddpg import DDPG
from envs.DirectControllerv2 import DirectControllerPT2
import numpy as np
import matplotlib.pyplot as plt


model = DDPG.load(r"C:\AktiveProjekte\Python\Projektmodul\eval_final\reward_function\DDPG-gain_0.001-oscillation_fun_square-error_fun_no_fun\model.zip")

env = DirectControllerPT2(log=True)

t = np.linspace(0, 1.5, 15000)
sine = np.sin(2 * np.pi * 4 * t + np.pi/2)

obs = env.reset(custom_w=sine)
done = False
env.sim.last_state = [[0, sine[0] * 1000]]
while not done:
    action, _states = model.predict(obs)
    obs, rew, done, _ = env.step(action)

a = np.linspace(0, 1.5, 150)

plt.plot(t, sine)
plt.plot(t, env.episode_log["function"]["y"])
plt.plot(a, env.episode_log["action"]["value"])
plt.show()
