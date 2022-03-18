from stable_baselines3 import DDPG
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from envs.DirectControllerOnline import DirectControllerOnline

online_connection = DirectControllerOnlineConnection()
env = DirectControllerOnline(online_connection, output_freq=200)



model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\working_online_agents\1\end_model2.zip", env)


while True:
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        # action = [0]
        obs, reward, done, info = env.step(action)
        # done = False


