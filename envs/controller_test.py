from envs.DirectControllerPT2 import DirectControllerPT2
import matplotlib.pyplot as plt
import numpy as np



env = DirectControllerPT2()

obs = env.reset()

print(300/500)

print(env.sys_gain)
print(env.sim.action_scale)
print(env.sim.obs_scale)

print(env.sim.obs_scale/(env.sys_gain * env.sim.action_scale))


print(env.w[0] * env.sim.obs_scale/(env.sys_gain * env.sim.action_scale))
print(obs)