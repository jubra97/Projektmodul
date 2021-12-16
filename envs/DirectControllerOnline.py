import time

import gym
from envs.OnlineSystem import OnlineSystem
import copy
import numpy as np


class DirectControllerOnline(gym.Env):

    def __init__(self, sample_freq=5000, log=True):
        super(DirectControllerOnline).__init__()

        self.online_system = OnlineSystem()

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []
        self.n_episodes = 0
        self.sample_freq = sample_freq
        self.log = log

        self.t = []
        self.w = []
        self.u = []
        self.y = []

        self.last_step_time = None

        self.integrated_error = 0
        self.abs_integrated_error = 0


        self.observation_space = gym.spaces.Box(low=np.array([-100]*6), high=np.array([100]*6),
                                                shape=(6,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

    def reset(self):
        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1
        print(self.n_episodes)

        self.online_system.reset()

        self.integrated_error = 0
        self.abs_integrated_error = 0


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

        obs = self.create_obs()
        return obs

    def step(self, action):

        if self.last_step_time:
            print(time.perf_counter() - self.last_step_time)
        self.last_step_time = time.perf_counter()

        u = (action[0] + 1) * 250
        u = np.clip(u, 0, 500)

        self.update_input()
        obs = self.create_obs()
        reward = self.create_reward()
        self.online_system.set_u(u)

        # when is epside done? Just time?
        info = {}

        if self.t[-1] > 20:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def update_input(self):
        with self.online_system.ads_buffer_mutex:
            self.t = copy.copy(self.online_system.last_t[-2:])
            self.w = copy.copy(self.online_system.last_w[-2:])
            self.u = copy.copy(self.online_system.last_u[-2:])
            self.y = copy.copy(self.online_system.last_y[-2:])

    def create_reward(self):
        y = np.array(self.y)
        w = np.array(self.w)
        e = np.mean(w - y)
        self.integrated_error = self.integrated_error + e * (1/self.online_system.sample_freq)
        self.integrated_error = np.clip(self.integrated_error, -0.3, 0.3)
        self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.online_system.sample_freq)
        self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)


        pen_error = np.abs(e)
        pen_error = np.sqrt(pen_error) * 1
        pen_action = np.abs(self.u[-2] - self.u[-1])

        pen_action = np.sqrt(pen_action) * 0.1
        pen_integrated = np.square(self.integrated_error) * 50

        reward = pen_error + pen_action + pen_integrated
        reward = -reward*5

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)
            self.episode_log["rewards"]["pen_error_integrated"].append(-pen_integrated)

        return reward

    def create_obs(self):
        if not self.w:
            return [0, 0, 0, 0, 0, 0]

        set_pont = self.w[-1]
        set_point_vel = (self.w[-1] - self.w[-2]) * 1 / self.sample_freq
        system_input = self.u[-1]
        system_input_vel = (self.u[-1] - self.u[-2]) * 1 / self.sample_freq
        system_output = self.y[-1]
        system_output_vel = (self.y[-1] - self.y[-2]) * 1 / self.sample_freq

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