import numpy as np
import matplotlib.pyplot as plt
from envs.DirectControllerOnline import DirectControllerOnline
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection

conn = DirectControllerOnlineConnection()
env = DirectControllerOnline(conn)

obs = env.reset()
print(obs)
done = False

u = np.linspace(-0.9, -0.6, 1000)
i = 0
while not done:
    obs, rew, done, info = env.step([u[i]])
    i += 1
    print(f"OBS: {obs}")
    print(f"REW: {rew}")