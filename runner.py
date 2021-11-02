import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import gym_test


# create DmsSim Gym Env
env = gym_test.DmsSim()

# use action and param noise?
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# use DDPG and create a tensorboard
# start tensorboard server with tensorboard --logdir ./dmsSim_ddpg_tensorboard/
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./dmsSim_ddpg_tensorboard/")
model.learn(total_timesteps=2000)

# save model if you want to
# model.save("dms_ddpg")
#
# del model # remove to demonstrate saving and loading

# load model if you want to
# model = DDPG.load("dms_ddpg)

# simulate solutions
while True:
    obs = env.reset()
    print(obs)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()