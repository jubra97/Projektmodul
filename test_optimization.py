import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import SimulationEnvs
from scipy.optimize import least_squares

env = SimulationEnvs.NoControllerAdaptivePT2()

p0 = np.ones(150) * 0.01
lb = np.ones(150) * -1
ub = np.ones(150)


def opt_fun(opt):
    rewards = []
    env.reset()
    for n in range(0, 150):
        obs, reward, dome, _ = env.step(np.array([opt[n]]))
        rewards.append(reward)
    rewards = np.array(rewards)
    return -rewards


out = least_squares(opt_fun, p0, bounds=(lb, ub), verbose=2)
print(out)

print(out.x)