from stable_baselines3 import DDPG
from rl.envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from rl.envs.DirectControllerOnline import DirectControllerOnline

online_connection = DirectControllerOnlineConnection()
env = DirectControllerOnline(online_connection, output_freq=100)


model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\working_online_agents\1\end_model2.zip", env)

import time
start = time.time()
obs = env.reset()
while True:

    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if time.time() - start > 5:
        env.online_system.reset()
        start = time.time()


# while True:
#     done = False
#     obs = env.reset()
#     while not done:
#         action, _ = model.predict(obs)
#         # action = [0]
#         obs, reward, done, info = env.step(action)
#         # done = False


