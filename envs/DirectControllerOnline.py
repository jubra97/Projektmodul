import time

import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
from envs.DirectControl import DirectController


class DirectControllerOnline(DirectController):
    def __init__(self, online_sys=None, sensor_freq=4000, log=True):
        """
        Create a gym environment to directly control the actuating value (u) the given online system.
        :param online_sys: Connection to the online system.
        :param sample_freq:
        :param log: Log data?
        """

        self.online_system = online_sys

        self.timesteps_last_episode = 0

        self.online_sys_needs_reset = True
        self.last_step_time = None
        self.last_reset_time = None

        super().__init__(log=log)

    def custom_reset(self):
        """
        Reset episode.
        :return:
        """
        # print time between two reset calls
        if self.last_reset_time is not None:
            print(time.perf_counter() - self.last_reset_time)
        self.last_reset_time = time.perf_counter()

        print(f"Episode: {self.n_episodes}")
        print(f"Timesteps: {self.timesteps_last_episode}")
        self.timesteps_last_episode = 0

        # Reset is done after final step of episode and not before the first stop of the next episode. So you have to
        # set a flag to reset the online system in the first learning step.
        self.online_sys_needs_reset = True

        self.last_step_time = None

        # get first data from online system and create observation with it
        self.update_input()

    def step(self, action):
        """
        Update u of online system, get achieved reward and create next observation.
        :param action: Between -1 and 1; is scaled according to online system.
        :return:
        """
        # reset system if needed
        if self.online_sys_needs_reset:
            self.online_system.reset()
            self.online_sys_needs_reset = False
            print("OnlineSys reset called")

        # count calls per episodes for debugging
        self.timesteps_last_episode += 1

        # wait here to set u with a specific sample time
        if self.last_step_time:
            while time.perf_counter() - self.last_step_time < (1 / self.output_freq):  # sample time: 100Hz
                ...  # busy waiting is necessary as non busy waiting is not precised enough
        self.last_step_time = time.perf_counter()

        u = action[0]
        # do logging
        if self.log:
            if self.last_u is not None and u is not None:
                self.episode_log["action"]["change"].append(u - self.last_u[-1])
            else:
                self.episode_log["action"]["change"].append(0)
            if u is not None:
                self.episode_log["action"]["value"].append(u)
            else:
                self.episode_log["action"]["value"].append(0)

        # get newest data from online system; create obs and reward and set u
        self.update_input()
        obs = self.observation_function()
        reward = self.reward_function()
        self.online_system.set_u(u)

        if self.last_t[-1] > 5 and self.timesteps_last_episode > 1:  # one episode is 3 seconds long
            done = True
            if self.log:
                self.episode_log["function"]["y"] = self.online_system.last_y
                self.episode_log["function"]["w"] = self.online_system.last_w
        else:
            done = False

        info = {}
        return obs, reward, done, info

    def update_input(self):
        """
        Update values for observation/reward with newest online values.
        :return:
        """
        retry = True
        while retry:
            with self.online_system.ads_buffer_mutex:
                if self.last_t[-1] != 0 or self.n_episodes == 1 or self.timesteps_last_episode == 0:
                    retry = False
                try:
                    self.last_t.extend(self.online_system.last_t[-self.measurements_per_output_update:])
                    self.last_w.extend(self.online_system.last_w[-self.measurements_per_output_update:])
                    self.last_u.extend(self.online_system.last_u[-self.measurements_per_output_update:])
                    self.last_y.extend(self.online_system.last_y[-self.measurements_per_output_update:])
                except IndexError:
                    ...


    # def create_reward(self, current_u):
    #     """
    #     Compute reward according to current u and system state.
    #     :param current_u: U
    #     :return:
    #     """
    #
    #     # only possible if online data is available
    #     if self.y.size < 2:
    #         return 0
    #
    #     # compute control error
    #     y = np.array(self.y)
    #     w = np.array(self.w)
    #     e = np.mean(w - y)
    #
    #     # add a penalty for changing u to achieve a non oscillating u / system
    #     if self.last_u is None or current_u is None:
    #         pen_action = 0
    #     else:
    #         action_change = abs(current_u - self.last_u)
    #         pen_action = np.sqrt(action_change) * 3
    #
    #     abs_error = abs(e)
    #     pen_error = np.abs(e)  # add penalty for high error between y and w
    #
    #     # add a reward for achieving smaller errors. Results in better training.
    #     reward = 0
    #     if abs_error < 0.5:
    #         reward += 1
    #     if abs_error < 0.1:
    #         reward += 2
    #     if abs_error < 0.05:
    #         reward += 3
    #     if abs_error < 0.02:
    #         reward += 4
    #     if abs_error < 0.01:
    #         reward += 5
    #     if abs_error < 0.005:
    #         reward += 10
    #
    #     reward -= pen_error
    #     reward -= pen_action
    #
    #     if self.log:
    #         self.episode_log["rewards"]["summed"].append(reward)
    #         self.episode_log["rewards"]["pen_error"].append(-pen_error)
    #         self.episode_log["rewards"]["pen_action"].append(-pen_action)
    #
    #     return reward
    #
    # def create_obs(self, current_u):
    #     """
    #     Create Observation consistent of [set_point (w), system_input (u), system_output (y), dot_w, dot_u, dot_y]
    #     :param current_u: Current u (action)
    #     :return:
    #     """
    #     if self.w.size < 2:
    #         print("Obs wrong")
    #         print(self.w)
    #         return [0, 0, 0, 0]
    #
    #     set_point = self.w[-1]
    #     set_point_vel = (self.w[-1] - self.w[-2]) * 1 / self.sample_freq
    #     system_output = self.y[-1]
    #     system_output_vel = (self.y[-1] - self.y[-2]) * 1 / self.sample_freq
    #     system_input = (self.u[-1] / 250) - 1
    #     error = self.w[-1] - self.y[-1]
    #     error_vel = ((self.w[-1] - self.y[-1]) - (self.w[-2] - self.y[-2])) * 1 / self.sample_freq
    #
    #     if self.last_u is None or current_u is None:
    #         input_vel = 0
    #     else:
    #         input_vel = (current_u - self.last_u) * 1 / self.sample_freq
    #
    #     # self.episode_log["obs"]["set_point"].append(set_point)
    #     # self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
    #     # self.episode_log["obs"]["system_output"].append(system_output)
    #     # self.episode_log["obs"]["system_output_vel"].append(system_output_vel)
    #     # self.episode_log["obs"]["system_input"].append(system_input)
    #     # self.episode_log["obs"]["input_vel"].append(input_vel)
    #     #
    #     # obs = [set_point, system_input, system_output, set_point_vel, input_vel, system_output_vel]
    #     # obs[2] = 0
    #     # obs[-1] = 0
    #
    #     self.episode_log["obs"]["error"].append(error)
    #     self.episode_log["obs"]["error_vel"].append(error_vel)
    #     self.episode_log["obs"]["system_input"].append(system_input)
    #     self.episode_log["obs"]["input_vel"].append(input_vel)
    #
    #     obs = [error, system_input, error_vel, input_vel]
    #
    #
    #     return np.array(obs)
    #
    # def render(self, mode="human"):
    #     pass
    #
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