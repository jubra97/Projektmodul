import time

import numpy as np
import matplotlib.pyplot as plt
from rl.envs.DirectControl import DirectController


class DirectControllerOnline(DirectController):
    def __init__(self,
                 online_sys=None,
                 log=True,
                 output_freq=100,
                 sensor_freq=4000,
                 reward_kwargs=None,
                 observation_kwargs=None,
                 ):
        """
        Create a gym environment to directly control the actuating value (u) the given online system.
        :param online_sys: Connection to the online system.
        :param log: Log data?
        :param output_freq: Frequency for u update.
        :param sensor_freq: Frequency of new sensor update data.
        :param reward_kwargs: Dict with extra options for the reward function
        :param observation_kwargs: Dict with extra option for the observation function
        """

        self.online_system = online_sys

        self.timesteps_last_episode = 0

        self.online_sys_needs_reset = True
        self.last_step_time = None
        self.last_reset_time = None

        super().__init__(log=log,
                         output_freq=output_freq,
                         sensor_freq=sensor_freq,
                         reward_kwargs=reward_kwargs,
                         observation_kwargs=observation_kwargs,
                         )

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

        u = self.last_u[-1] + (action[0])
        u = np.clip(u, -1, 1)
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
        self.online_system.set_u(u)
        self.update_input()
        obs = self.observation_function()
        reward = self.reward_function()

        if self.last_t[-1] > 5 and self.timesteps_last_episode > 1:  # one episode is 5 seconds long
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
        # try update data until mutex is free and data some data is already written
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
        ax[1][0].plot([0], [0], label=f"Sum: {np.sum(self.episode_log['rewards']['summed']):.2f}")
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

        rmse_episode = np.mean(np.sqrt(np.square(np.array(self.episode_log["function"]["w"]) - np.array(self.episode_log["function"]["y"]))))

        fig.tight_layout()
        return fig, ax, rmse_episode, None