from stable_baselines3 import DDPG
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
from envs.DirectControllerOnline import DirectControllerOnline

online_connection = DirectControllerOnlineConnection()
env = DirectControllerOnline(online_connection)

model = DDPG.load(r"C:\Users\brandlju\PycharmProjects\Projektmodul\eval\direct_with_error\end_model.zip", env)


while True:
    done = False
    obs = env.reset()
    while not done:
        obs[0] = 0
        obs[1] = 0
        action, _ = model.predict(obs)
        if len(env.t) > 0:
            print(env.t[-1])
        print(obs)
        print(action)
        print("______")
        # action = [0]
        obs, reward, done, info = env.step(action)
        # done = False


