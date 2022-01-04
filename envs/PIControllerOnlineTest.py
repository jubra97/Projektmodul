import numpy as np

from envs.PIControllerOnline import PIControllerOnline
import matplotlib.pyplot as plt

env = PIControllerOnline(log=True)

obs = env.reset()
done = False
p = np.linspace(-1, 1, 20 * 250)
run = 0
while not done:
    obs, reward, done, _ = env.step([p[run], p[run]])
    run += 1
print(run)
env.create_eval_plot()
plt.show()