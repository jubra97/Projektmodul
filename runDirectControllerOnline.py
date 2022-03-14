from stable_baselines3 import DDPG
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from envs.DirectControllerOnline import DirectControllerOnline

online_connection = DirectControllerOnlineConnection()
env = DirectControllerOnline(online_connection)

model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\eval\abc\end_model.zip", env)


while True:
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        # action = [0]
        obs, reward, done, info = env.step(action)
        # done = False


