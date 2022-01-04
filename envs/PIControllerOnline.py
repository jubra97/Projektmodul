import time

import gym
from envs.OnlineSystemPI import OnlineSystemPI
import copy
import numpy as np
import matplotlib.pyplot as plt

class PIControllerOnline(gym.Env):

    def __init__(self, sample_freq=5000, log=True):
        super(PIControllerOnline).__init__()

        self.online_system = OnlineSystemPI()

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []
        self.n_episodes = 0
        self.sample_freq = sample_freq
        self.log = log

        self.t = []
        self.w = []
        self.u = []
        self.y = []
        self.last_p = 0
        self.last_i = 0

        self.last_step_time = None

        self.last_reset_call = None
        self.integrated_error = 0
        self.abs_integrated_error = 0


        self.observation_space = gym.spaces.Box(low=np.array([-100]*6), high=np.array([100]*6),
                                                shape=(6,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]*2), high=np.array([1]*2), shape=(2,),
                                           dtype=np.float32)

    def reset(self):
        if self.last_reset_call is not None:
            print(time.perf_counter() - self.last_reset_call)
        self.last_reset_call = time.perf_counter()

        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1
        print(self.n_episodes)

        self.online_system.reset()

        self.integrated_error = 0
        self.abs_integrated_error = 0
        self.last_step_time = 0

        self.episode_log["obs"]["last_p"] = []
        self.episode_log["obs"]["last_i"] = []
        self.episode_log["obs"]["set_point"] = []
        self.episode_log["obs"]["set_point_vel"] = []
        self.episode_log["obs"]["system_output"] = []
        self.episode_log["obs"]["system_output_vel"] = []
        self.episode_log["rewards"]["summed"] = []
        self.episode_log["rewards"]["pen_error"] = []
        self.episode_log["rewards"]["pen_u_change"] = []
        self.episode_log["rewards"]["pen_error_integrated"] = []
        self.episode_log["action"]["value"] = []
        self.episode_log["action"]["change"] = []
        self.episode_log["function"]["w"] = []
        self.episode_log["function"]["y"] = []

        obs = self.create_obs()
        return obs

    def step(self, action):

        if self.last_step_time:
            while time.perf_counter() - self.last_step_time < 0.004:
                ...
                time.sleep(0.00001)
            # print(time.perf_counter() - self.last_step_time)
        self.last_step_time = time.perf_counter()


        p = (action[0] + 1) * 1.9 + 0.1
        p = np.clip(p, 0.1, 2)

        i = (action[1] + 1) * 0.25
        i = np.clip(p, 0, 0.5)

        self.update_input()
        obs = self.create_obs()
        reward = self.create_reward()
        self.online_system.set_pi(p, i)

        self.last_p = p
        self.last_i = i

        # when is epside done? Just time?
        info = {}

        if self.t and self.t[-1] > 20:
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
        u = np.array(self.u)
        e = np.mean(w - y)

        self.integrated_error = self.integrated_error + e * (1/self.online_system.sample_freq)
        self.integrated_error = np.clip(self.integrated_error, -0.3, 0.3)
        self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.online_system.sample_freq)
        self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)

        u_change = np.mean(np.diff(u))

        pen_error = np.abs(e)
        pen_u_change = np.abs(u_change) * 0.05

        # pen_error = np.sqrt(pen_error) * 1
        # pen_action = np.abs(self.u[-2] - self.u[-1])

        # pen_action = np.sqrt(pen_action) * 0.1
        # pen_integrated = np.square(self.integrated_error) * 50

        reward = -pen_error - pen_u_change
        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_u_change"].append(-pen_u_change)
            # self.episode_log["rewards"]["pen_error_integrated"].append(-pen_integrated)

        return reward

    def create_obs(self):
        if not self.w or len(self.w) < 2:
            return [0, 0, 0, 0, 0, 0]

        set_pont = self.w[-1]
        set_point_vel = (self.w[-1] - self.w[-2]) * 1 / self.sample_freq
        system_output = self.y[-1]
        system_output_vel = (self.y[-1] - self.y[-2]) * 1 / self.sample_freq

        self.episode_log["obs"]["last_p"].append(self.last_p)
        self.episode_log["obs"]["last_i"].append(self.last_i)
        self.episode_log["obs"]["set_point"].append(set_pont)
        self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
        self.episode_log["obs"]["system_output"].append(system_output)
        self.episode_log["obs"]["system_output_vel"].append(system_output_vel)

        obs = [self.last_p, self.last_i, set_pont, set_point_vel, system_output, system_output_vel]
        return obs

    def render(self, mode="human"):
        pass

    def create_eval_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(20, 12))

        ax[0][0].set_title("Obs")
        for key, value in self.episode_log["obs"].items():
            if len(value) > 0:
                ax[0][0].plot(value[:-1], label=key)  # verschoben zum Reset; eine Obs mehr als Reward!
        ax[0][0].grid()
        ax[0][0].legend()

        ax[1][0].set_title("Rewards")
        for key, value in self.episode_log["rewards"].items():
            if len(value) > 0:
                ax[1][0].plot(value, label=key)
        ax[1][0].grid()
        ax[1][0].legend()

        ax[0][1].set_title("Action")
        for key, value in self.episode_log["action"].items():
            if len(value) > 0:
                ax[0][1].plot(value, label=key)
        ax[0][1].grid()
        ax[0][1].legend()

        ax[1][1].set_title("Function")
        for key, value in self.episode_log["function"].items():
            if len(value) > 0:
                ax[1][1].plot(value, label=key)
        ax[1][1].grid()
        ax[1][1].legend()

        fig.tight_layout()
        return fig