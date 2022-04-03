from envs.DirectControllerSim import DirectControllerSim
import numpy as np

env = DirectControllerSim()
observations = []
rewards = []
states = []

actions = [0] * 50 + [0.0098] * 25 + [0.008] + [0] * 124

obs = env.reset(step_start=0, step_end=0.5, step_slope=0.25)
print(obs)
done = False
i = 0
while not done:
    obs, reward, done, _ = env.step([actions[i]])
    rewards.append(reward)
    observations.append(obs)
    states.append(env.sim.last_state[:, -1])
    i += 1

import matplotlib.pyplot as plt

plt.plot(observations)
plt.show()

plt.plot(rewards)
plt.show()
print(np.sum(rewards))
plt.plot(env.w)
plt.plot(env.sim._sim_out)
plt.show()

# plt.plot(env.sim._sim_out[2, :])
# plt.show()

# plt.plot(states)
# plt.show()