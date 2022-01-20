import time

import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

class DirectControllerOnline(gym.Env):
    def __init__(self, online_sys=None, sample_freq=5000, log=True):
        super(DirectControllerOnline).__init__()

        self.online_system = online_sys

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []
        self.n_episodes = 0
        self.timesteps_last_episode = 0
        self.sample_freq = sample_freq
        self.log = log
        self.online_sys_needs_reset = True

        self.t = np.array([])
        self.w = np.array([])
        self.u = np.array([])
        self.y = np.array([])
        self.last_u = None

        self.last_step_time = None

        self.last_reset_call = None
        self.integrated_error = 0
        self.abs_integrated_error = 0

        self.observation_space = gym.spaces.Box(low=np.array([-100]*6), high=np.array([100]*6),
                                                shape=(6,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]*1), high=np.array([1]*1), shape=(1,),
                                           dtype=np.float32)

    def reset(self):
        ###
        # Reset is done after final step and before learning. Reset is not called after learning!
        ###

        if self.last_reset_call is not None:
            print(time.perf_counter() - self.last_reset_call)
        self.last_reset_call = time.perf_counter()

        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1
        print(f"Episode: {self.n_episodes}")
        print(f"Timesteps: {self.timesteps_last_episode}")
        self.online_sys_needs_reset = True

        self.timesteps_last_episode = 0

        self.integrated_error = 0
        self.abs_integrated_error = 0
        self.last_step_time = 0
        self.last_u = None

        self.episode_log["obs"]["set_point"] = []
        self.episode_log["obs"]["set_point_vel"] = []
        self.episode_log["obs"]["system_output"] = []
        self.episode_log["obs"]["system_output_vel"] = []
        self.episode_log["obs"]["system_input"] = []
        self.episode_log["obs"]["input_vel"] = []
        self.episode_log["rewards"]["summed"] = []
        self.episode_log["rewards"]["pen_error"] = []
        self.episode_log["rewards"]["pen_u_change"] = []
        self.episode_log["rewards"]["pen_error_integrated"] = []
        self.episode_log["rewards"]["pen_action"] = []
        self.episode_log["action"]["value"] = []
        self.episode_log["action"]["change"] = []
        self.episode_log["function"]["w"] = []
        self.episode_log["function"]["y"] = []

        obs = self.create_obs(None)
        return obs

    def step(self, action):
        if self.online_sys_needs_reset:
            self.online_system.reset()
            self.online_sys_needs_reset = False

        self.timesteps_last_episode += 1

        if self.last_step_time:
            while time.perf_counter() - self.last_step_time < 0.01:
                ...
                # time.sleep(0.00001)
            # print(time.perf_counter() - self.last_step_time)
        self.last_step_time = time.perf_counter()

        u = (action[0] + 1) * 250
        u = np.clip(u, 0, 500)


        if self.last_u is not None and u is not None:
            self.episode_log["action"]["change"].append((u - (self.last_u+1) * 250))
        else:
            self.episode_log["action"]["change"].append(0)
        if u is not None:
            self.episode_log["action"]["value"].append(u)
        else:
            self.episode_log["action"]["value"].append(0)


        self.update_input()
        obs = self.create_obs(action[0])
        reward = self.create_reward(action[0])
        self.last_u = action[0]
        self.online_system.set_u(u)

        # when is epside done? Just time?
        info = {}

        if self.t is not None and self.t.size > 0 and self.t[-1] > 3:
            done = True
            self.episode_log["function"]["y"] = self.online_system.last_y
            self.episode_log["function"]["w"] = self.online_system.last_w

        else:
            done = False

        return obs, reward, done, info

    def update_input(self):
        with self.online_system.ads_buffer_mutex:
            self.t = np.array(copy.copy(self.online_system.last_t[-2:]))
            self.w = np.array(copy.copy(self.online_system.last_w[-2:])) / 3_000_000
            self.u = np.array(copy.copy(self.online_system.last_u[-2:]))
            self.y = np.array(copy.copy(self.online_system.last_y[-2:])) / 3_000_000

    def create_reward(self, current_u):
        if self.y.size > 0:
            y = np.array(self.y)
            w = np.array(self.w)
            e = np.mean(w - y)
        else:
            return 0

        if self.last_u == None:
            action_change = 0
            pen_action = 0
        else:
            action_change = abs(current_u - self.last_u)
            pen_action = np.sqrt(action_change) * 3

        # self.integrated_error = self.integrated_error + e * (1/self.online_system.sample_freq)
        # self.integrated_error = np.clip(self.integrated_error, -0.3, 0.3)
        # self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.online_system.sample_freq)
        # self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)

        if self.u is None:
            return 0

        abs_error = abs(e)
        pen_error = np.abs(e)


        reward = 0
        if abs_error < 0.5:
            reward += 1
        if abs_error < 0.1:
            reward += 2
        if abs_error < 0.05:
            reward += 3
        if abs_error < 0.02:
            reward += 4
        if abs_error < 0.01:
            reward += 5
        if abs_error < 0.005:
            reward += 10

        reward -= pen_error
        reward -= pen_action
        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)

        return reward

    def create_obs(self, current_u):
        if self.w.size < 2:
            return [0, 0, 0, 0, 0, 0]

        set_point = self.w[-1]
        set_point_vel = (self.w[-1] - self.w[-2]) * 1 / self.sample_freq
        system_output = self.y[-1]
        system_output_vel = (self.y[-1] - self.y[-2]) * 1 / self.sample_freq
        system_input = (self.u[-1] / 250) - 1
        if self.last_u is None:
            input_vel = 0
        else:
            input_vel = (current_u - self.last_u) * 1 / self.sample_freq

        self.episode_log["obs"]["set_point"].append(set_point)
        self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
        self.episode_log["obs"]["system_output"].append(system_output)
        self.episode_log["obs"]["system_output_vel"].append(system_output_vel)
        self.episode_log["obs"]["system_input"].append(system_input)
        self.episode_log["obs"]["input_vel"].append(input_vel)

        obs = [set_point, system_input, system_output, set_point_vel, input_vel, system_output_vel]

        return np.array(obs)

    def render(self, mode="human"):
        pass

    def create_eval_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(20, 12))

        ax[0][0].set_title("Obs")
        for key, value in self.episode_log["obs"].items():
            if value:
                ax[0][0].plot(value[:-1], label=key)  # verschoben zum Reset; eine Obs mehr als Reward!
        ax[0][0].grid()
        ax[0][0].legend()

        ax[1][0].set_title("Rewards")
        for key, value in self.episode_log["rewards"].items():
            if value:
                ax[1][0].plot(value, label=key)
        ax[1][0].grid()
        ax[1][0].legend()

        ax[0][1].set_title("Action")
        for key, value in self.episode_log["action"].items():
            if value:
                ax[0][1].plot(value, label=key)
        ax[0][1].grid()
        ax[0][1].legend()

        ax[1][1].set_title("Function")
        for key, value in self.episode_log["function"].items():
            if value:
                ax[1][1].plot(value, label=key)
        ax[1][1].grid()
        ax[1][1].legend()

        fig.tight_layout()
        return fig