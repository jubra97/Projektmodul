import gym
from envs.OnlineSystem import OnlineSystem
import copy
import numpy as np


class DirectControllerOnline(gym.Env):

    def __init__(self, sample_freq=5000):
        super(DirectControllerOnline).__init__()

        self.online_system = OnlineSystem()

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []
        self.n_episodes = 0
        self.sample_freq = sample_freq


        self.observation_space = gym.spaces.Box(low=np.array([-100]*7), high=np.array([100]*7),
                                                shape=(7,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

    def reset(self):
        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1


        self.episode_log["obs"]["last_set_points"] = []
        self.episode_log["obs"]["last_system_inputs"] = []
        self.episode_log["obs"]["last_system_outputs"] = []
        self.episode_log["obs"]["errors"] = []
        self.episode_log["obs"]["set_point"] = []
        self.episode_log["obs"]["system_input"] = []
        self.episode_log["obs"]["system_output"] = []
        self.episode_log["obs"]["set_point_vel"] = []
        self.episode_log["obs"]["input_vel"] = []
        self.episode_log["obs"]["outputs_vel"] = []
        self.episode_log["obs"]["integrated_error"] = []
        self.episode_log["rewards"]["summed"] = []
        self.episode_log["rewards"]["pen_error"] = []
        self.episode_log["rewards"]["pen_action"] = []
        self.episode_log["rewards"]["pen_error_integrated"] = []
        self.episode_log["action"]["value"] = []
        self.episode_log["action"]["change"] = []
        self.episode_log["function"]["w"] = None
        self.episode_log["function"]["y"] = None

        self.create_obs()
        return obs

    def step(self, action):
        return obs, reward, done, info


    def create_reward(self):

    def create_obs(self):
        with self.online_system.ads_buffer_mutex:
            w = copy.copy(self.online_system.last_w[-2:])
            u = copy.copy(self.online_system.last_u[-2:])
            y = copy.copy(self.online_system.last_y[-2:])

        set_pont = w[-1]
        set_point_vel = (w[-1] - w[-2]) * 1 / self.sample_freq
        system_input = u[-1]
        system_input_vel = (u[-1] - u[-2]) * 1 / self.sample_freq
        system_output = y[-1]
        system_output_vel = (y[-1] - y[-2]) * 1 / self.sample_freq

        self.episode_log["obs"]["set_point"] = set_pont
        self.episode_log["obs"]["system_input"] = system_input
        self.episode_log["obs"]["system_output"] = system_output
        self.episode_log["obs"]["set_point_vel"] = set_point_vel
        self.episode_log["obs"]["input_vel"] = system_input_vel
        self.episode_log["obs"]["outputs_vel"] = system_output_vel

        obs = [set_pont, set_point_vel, system_input, system_input_vel, system_output, system_output_vel]
        return obs


    def render(self, mode="human"):
        pass