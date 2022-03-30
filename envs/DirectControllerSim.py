import json
import pathlib
from collections import deque

import control
import matplotlib.pyplot as plt
import numpy as np

from Simulation.OpenLoopSim import OpenLoopSim
from envs.DirectControl import DirectController


class DirectControllerSim(DirectController):
    def __init__(self,
                 log=False,
                 model_freq=12_000,
                 output_freq=100,
                 sensor_freq=4000,
                 sys=None,
                 action_scale=None,
                 observation_scale=None,
                 reward_kwargs=None,
                 observation_kwargs=None,
                 ):
        """
        Create a DirectController with a simulation backend. This class can use the Observation and Reward functions of
        the DirectController. It uses the OpenLoopSim Class to simulate a SISO StateSpace or TransferFunction with
        Step-by-Step updated inputs. The input is computed from the old input + a change, which is the action.
        That action has to be learned by the RL algorithm.
        :param log: log env outcomes
        :param model_freq: Simulation update frequency.
        :param output_freq: Frequency for u update.
        :param sensor_freq: Frequency of new sensor update data.
        :param sys: System to be simulated. Use a pre-defined system if None.
        :param action_scale: How to scale the action in a way to fit to the system.
        :param observation_scale: How to scale the observation in a way to fit to the system.
        :param reward_kwargs: Dict with extra options for the reward function
        :param observation_kwargs: Dict with extra option for the observation function
        """
        if sys:
            assert action_scale, "Add a action scale factor for your system. If no scaling should be done use 1."
        else:
            sys = control.tf([3.452391113940120e+04], [1,46.629595161020170,3.495142304939126e+04])  # pt2 of dms
            action_scale = 1
            observation_scale = 1
        # create simulation object with an arbitrary tf.
        self.sim = OpenLoopSim(sys,
                               model_freq,
                               sensor_freq,
                               output_freq,
                               action_scale=action_scale,
                               obs_scale=observation_scale,
                               simulation_time=1.5)
        self.sys_gain = (self.sim.sys.C @ np.linalg.inv(-self.sim.sys.A) @ self.sim.sys.B + self.sim.sys.D)[0][0]
        self.w = None

        super().__init__(log=log,
                         sensor_freq=sensor_freq,
                         output_freq=output_freq,
                         reward_kwargs=reward_kwargs,
                         observation_kwargs=observation_kwargs,
                         )

    def custom_reset(self, step_start=None, step_end=None, step_slope=None, custom_w=None):
        """
        Reset the simulation. Create a new reference value w. With the input arguments you can chose a course for w.
        After that, set the initial state for the next simulation episode.
        :param step_start: Initial value before Step: If None it is chosen randomly between (0, 1)
        :param step_end: Final value after Step: If None it is chosen randomly between (0, 1)
        :param step_slope: Rising time in sec. If 0 it is a step. If > 0 w is a ramp.
        :param custom_w: Your own reference value.
        :return:
        """
        # reset simulation
        self.sim.reset()

        # create reference value (w)
        if custom_w is not None:
            assert self.sim.n_sample_points == len(custom_w), "Simulation and input length must not differ"
            self.w = custom_w
        else:
            self.w = self.set_w(step_start, step_end, step_slope)

        # set x0 in state space, to do so compute step response to given w[0] and set the system states accordingly.
        T = control.timeresp._default_time_vector(self.sim.sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        U = np.ones_like(T) * self.w[0] * (self.sim.obs_scale / self.sys_gain)
        _, step_response, states = control.forced_response(self.sim.sys, T, U, return_x=True)
        self.sim.last_state = np.array([states[:, -1]]).T

        # set deque with initial states
        initial_input = ((self.w[0] * self.sim.obs_scale) / (self.sys_gain * self.sim.action_scale))
        self.last_u = deque([initial_input] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        initial_output = (self.sim.last_state[:, -1] @ self.sim.sys.C.T)[0] / self.sim.obs_scale
        self.last_y = deque([initial_output] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([self.w[0]] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

    def set_w(self, step_start=None, step_end=None, step_slope=None):
        """
        Create reference value (w) as ramp or step
        :param step_start: Initial value before Step: If None it is chosen randomly between (0, 1)
        :param step_end: Final value after Step: If None it is chosen randomly between (0, 1)
        :param step_slope: Slope of ramp; if 0 a step is generated
        :return:
        """
        if step_start is None:
            step_start = np.random.uniform(0, 1)
        if step_end is None:
            step_end = np.random.uniform(0, 1)
        if step_slope is None:
            step_slope = np.random.uniform(0, 0.5)
        w_before_step = [step_start] * int(0.5 * self.sim.model_freq)
        w_step = np.linspace(w_before_step[0], step_end, int(step_slope * self.sim.model_freq)).tolist()
        w_after_step = [step_end] * int(self.sim.n_sample_points - len(w_before_step) - len(w_step))
        w = w_before_step + w_step + w_after_step
        return w

    def update_simulation(self, u_trajectory):
        """
        Sim one step of the simulation with u_trajectory as input. Update the deques that hold the measured data.
        :param u_trajectory: Input trajectory for simulation.
        :return: None
        """
        w_current_sim_step = self.w[self.sim.current_simulation_step + len(
            u_trajectory):self.sim.current_simulation_step + 1:-self.sim.model_steps_per_senor_update][::-1]
        # simulate system until next update of u.
        t, u, y = self.sim.sim_one_step(u=u_trajectory, add_noise=True)

        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time is used for more precise simulation results.
        # Sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # In this fifo lists the newest data is appended at the rights site.

        for step in range(len(t)):
            self.last_t.append(t[step])
            self.last_u.append(u[step])
            self.last_y.append(y[step])
            self.last_w.append(w_current_sim_step[step])

    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Change of u. action is always between (-1, 1)
        :return:
        """
        # use action[0] * 2 to be able to change the action from one extreme to the other.
        # If you want to change the system input from -1 to 1 in one step you need 2 as action.

        new_action = self.last_u[-1] + action[0] * 2
        new_action = np.clip(new_action, -1, 1)
        system_input_trajectory = [new_action] * (self.sim.model_steps_per_controller_update + 1)
        self.update_simulation(system_input_trajectory)

        if self.log:
            self.episode_log["action"]["value"].append(new_action)
            self.episode_log["action"]["change"].append(action[0])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = np.array(self.w) * self.sim.obs_scale
                self.episode_log["function"]["y"] = np.array(self.sim._sim_out) * self.sim.obs_scale
        else:
            done = False

        # create obs and reward
        obs = self.observation_function()
        reward = self.reward_function()
        return obs, reward, done, {}

    def create_eval_plot(self):
        """
        Crate a plot for tensorboard while training and afterwards for evaluation.
        :return:
        """
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        timestamps = np.linspace(0, self.last_t[-1], int(self.last_t[-1] * self.output_freq))

        ax[0][0].set_title("Obs")
        for key, value in self.episode_log["obs"].items():
            if len(value) > 0:
                ax[0][0].plot(timestamps, value[:-1], label=key)  # verschoben zum Reset; eine Obs mehr als Reward!
        ax[0][0].grid()
        ax[0][0].legend()

        ax[1][0].set_title("Rewards")
        for key, value in self.episode_log["rewards"].items():
            if len(value) > 0:
                ax[1][0].plot(timestamps, value, label=key)
        ax[1][0].plot([0], [0], label=f"Sum: {np.sum(self.episode_log['rewards']['summed']):.2f}")
        ax[1][0].grid()
        ax[1][0].legend()

        ax[0][1].set_title("Action")
        for key, value in self.episode_log["action"].items():
            if len(value) > 0:
                ax[0][1].plot(timestamps, value, label=key)
        ax[0][1].grid()
        ax[0][1].legend()

        ax[1][1].set_title("Function")
        for key, value in self.episode_log["function"].items():
            if len(value) > 0:
                ax[1][1].plot(self.sim.t, value, label=key)
        ax[1][1].grid()
        ax[1][1].legend()

        ax[1][2].set_title("FFT")
        actions_fft, actions_fftfreq, N, sm = self.eval_fft()
        ax[1][2].plot(actions_fftfreq, 2 / N * np.abs(actions_fft[0:N // 2]))
        ax[1][2].text(0.5, 0.9, f"Smoothness: {sm}", transform=ax[1][2].transAxes)
        ax[1][2].grid()

        fig.tight_layout()
        return fig, ax

    def eval(self, model, folder_name):
        """
        Run an evaluation with different steps and ramps. Create a plot for every run and save it. Also save a json file
        with some statistics of a run.
        :param model: Model to be used for action prediction.
        :param folder_name: Folder to save evaluation in.
        """
        steps = np.linspace(0, 0.5, 20)
        slopes = np.linspace(0, 0.5, 3)
        i = 1
        pathlib.Path(f"{folder_name}").mkdir(exist_ok=True)
        rewards = []
        rmse = []
        sms = []
        rise_times = []
        setting_times = []
        extra_info = {}
        for step in steps:
            for slope in slopes:
                # slope = slope * 0.1
                actions = []
                # create env
                done = False
                obs = self.reset(0, step, slope)
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, info = self.step(action)
                    rewards.append(reward)
                    actions.append(action)
                _, _, _, smoothness = self.eval_fft()
                sms.append(smoothness)
                fig, ax = self.create_eval_plot()

                np_sim_out = np.array(self.sim._sim_out)

                rmse_episode = np.sqrt(np.square(np.array(self.w) - np_sim_out))
                rmse.append(rmse_episode)

                if slope == 0 and step != 0:
                    # calculate rise time from 0.1 to 0.9 of step
                    rise_start = 0.1 * step
                    rise_stop = 0.9 * step
                    start_time = int(self.sim.model_freq * 0.5)
                    index_start = np.argmax(np_sim_out[start_time:] > rise_start)
                    index_end = np.argmax(np_sim_out[start_time:] > rise_stop)
                    rise_time = (index_end - index_start) / self.sim.model_freq
                    rise_times.append(rise_time)

                    # calculate setting time with 5% band
                    lower_bound = step - step * 0.05
                    upper_bound = step + step * 0.05
                    # go backwards through sim._sim_out and find first index out of bounds
                    index_lower = np.argmax(np_sim_out[::-1] < lower_bound)
                    index_upper = np.argmax(np_sim_out[::-1] > upper_bound)
                    last_out_of_bounds = min([index_lower, index_upper])
                    setting_time = (self.sim.n_sample_points - last_out_of_bounds - start_time) / self.sim.model_freq
                    setting_times.append(setting_time)

                    ax[0][2].text(0.1, 0.9, f"Rise Time: {rise_time}", transform=ax[0][2].transAxes)
                    ax[0][2].text(0.1, 0.7, f"Setting Time: {setting_time}", transform=ax[0][2].transAxes)
                ax[0][2].text(0.1, 0.5, f"Mean RMSE: {np.mean(rmse_episode)}", transform=ax[0][2].transAxes)
                plt.savefig(f"{folder_name}\\{i}_{step}_{slope}.png")
                plt.close()
                i += 1

        mean_episode_reward = np.sum(rewards) / self.n_episodes
        extra_info["mean_episode_reward"] = mean_episode_reward
        extra_info["rmse"] = np.mean(rmse)
        extra_info["max_rmse"] = np.max(rmse)
        extra_info["mean_rise_time"] = np.mean(rise_times)
        extra_info["mean_setting_time"] = np.mean(setting_times)
        extra_info["smoothness"] = np.mean(sms)

        with open(f"{folder_name}\\extra_info.json", 'w+') as f:
            json.dump(extra_info, f)
        print(f"Eval Info: RMSE: {np.mean(rmse)} --- Smoothness: {np.mean(sms)}")

        return extra_info