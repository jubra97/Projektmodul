import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import SimulationEnvs
from scipy.optimize import least_squares

env = SimulationEnvs.FullAdaptivePT2()

p0 = np.ones(300) * 0.0
lb = np.ones(300) * -1
ub = np.ones(300)


def opt_fun(opt, plot=False):
    rewards = []
    env.reset()
    for n in range(0, 150):
        obs, reward, dome, _ = env.step(np.array([opt[n], opt[n+150]]))
        rewards.append(reward)
    if plot:
        plt.plot(env.out)
        plt.plot(env.u)
        plt.show()
    rewards = np.array(rewards)
    rewards = -rewards
    # print(rewards[50:100])
    return rewards



out = least_squares(opt_fun, p0, bounds=(lb, ub), verbose=2, max_nfev=50)
print(out)

print(out.x)

# test = np.array([-0.99] * 300)
plt.plot(out.x[:150])
plt.plot(out.x[150:])
plt.show()
opt_fun(out.x, True)
