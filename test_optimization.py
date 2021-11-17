import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import SimulationEnvs
from scipy.optimize import least_squares

env = SimulationEnvs.NoControllerAdaptivePT2()

p0 = np.ones(300) * -0.95
lb = np.ones(300) * -1
ub = np.ones(300)


def opt_fun(opt, plot=False):
    rewards = []
    env.reset()
    for n in range(0, 150):
        obs, reward, dome, _ = env.step(np.array([opt[n], opt[n+150]]))
        rewards.append(reward)
    rewards = np.array(rewards)
    # rewards = -rewards
    if plot:
        plt.plot(rewards)
        plt.show()
        plt.plot(env.u)
        plt.plot(env.out)
        plt.show()
    return rewards


opt_fun(p0, True)

# out = least_squares(opt_fun, p0, bounds=(lb, ub), verbose=2, max_nfev=30)
# print(out)
#
# print(out.x)
# plt.plot(out.x[:150])
# plt.plot(out.x[150:])
# plt.show()
# opt_fun(out.x, True)