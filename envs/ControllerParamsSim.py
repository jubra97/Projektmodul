import json
import pathlib
from collections import deque

import control
import matplotlib.pyplot as plt
import numpy as np

from Simulation.IOSim import IOSim
from envs.ControllerParams import ControllerParams
from utils import custom_default


class ControllerParamsSim(ControllerParams):
    def __init__(self,
                 log=False,
                 model_freq=8_000,
                 output_freq=100,
                 sensor_freq=4000,
                 sys=None,
                 p_range=1,
                 i_range=1,
                 d_range=None,
                 reward_kwargs=None,
                 observation_kwargs=None,
                 ):
        """
        Create a gym environment for optimizing the PI(D) parameters of a PI(D) controller in every time step. Here the
        parameters are determined by simulating a SISO TF or SS system with them.
        :param log: Log the simulation outcomes.
        :param model_freq: Frequency of simulation update steps.
        :param output_freq: Frequency for u update.
        :param sensor_freq: Frequency of new sensor update data.
        :param sys: Open Loop system/plant to be simulated. Use a pre-defined system if None.
        :param reward_kwargs: Dict with extra options for the reward function
        :param observation_kwargs: Dict with extra option for the observation function
        """
        if sys:
            if isinstance(sys, control.TransferFunction):
                self.open_loop_sys = control.tf2ss(sys)
            else:
                self.open_loop_sys = sys
        else:
            self.open_loop_sys = control.tf2ss(control.tf([1], [0.003, 0.1, 1]))  # pt2 of dms
        # create simulation object with an arbitrary tf.
        self.sim = IOSim(None, model_freq, sensor_freq, output_freq, action_scale=1, obs_scale=1, simulation_time=1.5)
        # self.open_loop_gain = (self.open_loop_sys.C @ np.linalg.inv(-self.open_loop_sys.A) @ self.open_loop_sys.B + self.open_loop_sys.D)[0][0]
        self.open_loop_gain = 1
        super().__init__(log=log,
                         sensor_freq=sensor_freq,
                         output_freq=output_freq,
                         p_range=p_range,
                         i_range=i_range,
                         d_range=d_range,
                         reward_kwargs=reward_kwargs,
                         observation_kwargs=observation_kwargs
                         )

        self.first_step = True
        self.last_p = 1

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
            self.w = self.set_w(0, step_end, step_slope)
        self.first_step = True

        # set x0 in state space, to do so compute step response to given w[0]
        T = control.timeresp._default_time_vector(self.open_loop_sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        U = np.ones_like(T) * self.w[0, 0] * (self.sim.obs_scale / self.open_loop_gain)
        _, step_response, states = control.forced_response(self.open_loop_sys, T, U, return_x=True)

        extra_states = 0
        extra_states += 1 if "I" in self.controller_style else 0
        extra_states += 1 if "D" in self.controller_style else 0
        self.sim.last_state = np.concatenate((states[:, -1:], np.array([[0]] * extra_states)))

        initial_input = (self.w[0, 0] * self.sim.obs_scale)/(self.open_loop_gain * self.sim.action_scale)
        self.last_u = deque([initial_input] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        initial_output = (states[:, -1] @ self.open_loop_sys.C.T)[0] / self.sim.obs_scale
        self.last_y = deque([initial_output] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([self.w[0, 0]] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

        self.last_p = 1

    def set_w(self, step_start=None, step_end=None, step_slope=None):
        """
        Create reference value (w) as ramp or step. Also add noise as a second input to the system.
        :param step_start: Initial value before Step: If None it is chosen randomly between (0, 1)
        :param step_end: Final value after Step: If None it is chosen randomly between (0, 1)
        :param step_slope: Slope of ramp; if 0 a step is generated
        :return:
        """
        if step_start is None:
            step_start = np.random.uniform(0, 0.5)
        if step_end is None:
            step_end = np.random.uniform(0, 0.5)
        if step_slope is None:
            step_slope = np.random.uniform(0, 0.5)
        w_before_step = [step_start] * int(0.5 * self.sim.model_freq)
        w_step = np.linspace(w_before_step[0], step_end, int(step_slope * self.sim.model_freq)).tolist()
        w_after_step = [step_end] * int(self.sim.n_sample_points - len(w_before_step) - len(w_step))
        w = w_before_step + w_step + w_after_step

        noise = np.random.normal(0, 0.001, self.sim.n_sample_points)
        # noise = np.random.normal(0, 0.00, self.sim.n_sample_points)
        sys_input = np.array([w, noise])
        return sys_input

    def update_simulation(self):
        """
        Sim one step of the simulation with current closed loop system. Update the deques that hold the measured data.
        :return: None
        """
        w_current_sim_step = self.w[:, self.sim.current_simulation_step:self.sim.current_simulation_step+self.sim.model_steps_per_controller_update+1]
        # simulate system until next update of u.
        try:
            t, w, sim_out_sensor = self.sim.sim_one_step(w=w_current_sim_step)
        except RuntimeError as e:
            print(f"P: {self.last_p}")
            print(f"I: {self.last_i}")
            print(f"D: {self.last_d}")
            raise e
        u = sim_out_sensor[2, :]
        y = sim_out_sensor[1, :]
        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time is used for more precise simulation results.
        # Sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # For the fifo lists the newest sensor values are used.

        for step in range(len(t)):
            self.last_t.append(t[step])
            self.last_u.append(u[step])
            self.last_y.append(y[step])
            self.last_w.append(w[0, step])

    def step(self, action):
        """
        Step the environment for one output update step with the incoming action. Collect sensor values of the
        simulation and compute reward and next observation from them.
        :param action: Parameters for P, I, (D), as np.array. All those parameters must be between (-1, 1) and are
        scaled according to the scale parameters.
        :return: Observation, Reward, Done, Info
        """
        new_p = 0
        new_i = 0
        new_d = 0
        if "P" in self.controller_style:
            new_p = (action[0] + 1) * (self.p_range / 2)
            new_p = np.clip(new_p, 0, self.p_range)
        if "I" in self.controller_style:
            new_i = (action[1] + 1) * (self.i_range / 2)
            new_i = np.clip(new_i, 0, self.i_range)
        if "D" in self.controller_style:
            if len(self.controller_style):
                new_d = (action[2] + 1) * (self.d_range / 2)
                new_d = np.clip(new_d, 0, self.d_range)
            else:
                new_d = (action[1] + 1) * (self.d_range / 2)
                new_d = np.clip(new_d, 0, self.d_range)

        # system_input_trajectory = [new_action] * (self.sim.model_steps_per_controller_update + 1)
        self.sim.sys = self.create_closed_loop_io_sys(p=new_p, i=new_i, d=new_d)
        self.update_simulation()

        if self.log:
            self.episode_log["action"]["value_p"].append(new_p)
            self.episode_log["action"]["value_i"].append(new_i)
            # self.episode_log["action"]["change"].append(action[0])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = self.w[0, :]
                self.episode_log["function"]["y"] = self.sim._sim_out[1, :]
                self.episode_log["function"]["u"] = self.sim._sim_out[2, :]
                self.episode_log["function"]["e"] = self.sim._sim_out[3, :]
        else:
            done = False

        # create obs and reward
        obs = self.observation_function()
        reward = self.reward_function()
        return obs, reward, done, {}

    def create_closed_loop_io_sys(self, p, i=2, d=None):
        """
        Create a closed loop system based on the new pid parameters and the system plnat.
        :param p: P
        :param i: I
        :param d: D
        :return: closed loop system
        """

        # Set P and I to small very small values if they are 0 to prevent system from crashing.
        if p == 0:
            p = 1e-20
        if i == 0:
            i = 1e-20
        controller_p = control.tf([p], [1])
        if i is not None:
            controller_i = control.tf([i], [1, 0])
        else:
            controller_i = 0
        if d is not None:
            controller_d = control.tf([d, 0], [1e-9, 1])
        else:
            controller_d = control.tf([0], [1])

        controller = controller_p + controller_i + controller_d

        # p = p * self.sim.action_scale
        # i = i * self.sim.action_scale
        self.last_p = p
        self.last_i = i
        self.last_d = d
        # print(controller)
        # controller = control.tf2ss(controller)
        if self.first_step:
            self.sim.last_state[0, 0] = self.w[0, 0] / (i * self.open_loop_gain)
            self.first_step = False

        p_controller = control.LinearIOSystem(controller_p, inputs="e", outputs="u_p", name="controller_p")
        i_controller_gain = control.LinearIOSystem(control.tf([i], [1]), inputs="e", outputs="e_i", name="controller_i_gain")
        i_controller_int = control.LinearIOSystem(control.tf([1], [1, 0]), inputs="e_i", outputs="u_i", name="controller_i_int")
        i_controller = control.interconnect([i_controller_gain, i_controller_int], name="controller_i", inplist=["e"], inputs=["e"], outlist=["u_i"], outputs=["u_i"])
        d_controller = control.LinearIOSystem(controller_d, inputs="e", outputs="u_d", name="controller_d")
        controller_sum = control.summing_junction(inputs=["u_p", "u_i", "u_d"], output="u")
        controller = control.interconnect([p_controller, i_controller, d_controller, controller_sum], name="controller",
                                          inplist=["e"], outputs=["u"], inputs=["e"], outlist=["u"])

        io_open_loop = control.LinearIOSystem(self.open_loop_sys, inputs="u", outputs="y", name="open_loop")
        # io_controller = control.LinearIOSystem(controller, inputs="e", outputs="u", name="controller")
        w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
        y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

        closed_loop = control.interconnect([w_y_comp, io_open_loop, y_noise, controller],
                                           name="closed_loop",
                                           inplist=["w", "noise"],
                                           outlist=["y", "y_noisy", "u", "e"])
        # print(closed_loop.A)
        # print(closed_loop.B)
        # print(closed_loop.C)
        # print(closed_loop.D)
        return closed_loop#

    def create_eval_plot(self):
        """
        Crate a plot for tensorboard while training and afterwards for evaluation.
        :return:
        """
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        timestamps = np.linspace(0, self.last_t[-1], int(np.ceil(self.last_t[-1] * self.output_freq)))

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

        rmse_episode = np.mean(np.sqrt(np.square(np.array(self.w) - np.array(self.sim._sim_out[1, :]))))

        fig.tight_layout()
        return fig, ax, rmse_episode, sm

    def eval(self, model, folder_name, options_dict=None):
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

        for key, value in options_dict.items():
            extra_info[key] = value

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
                    index_end = np.argmax(np_sim_out[start_time + index_start:] > rise_stop)
                    rise_time = index_end / self.sim.model_freq
                    if rise_time == 0:
                        rise_time = 1
                    rise_times.append(rise_time)

                    # calculate setting time with 5% band
                    lower_bound = step - step * 0.05
                    upper_bound = step + step * 0.05
                    # go backwards through sim._sim_out and find first index out of bounds
                    index_lower_out = list(np.array(self.sim._sim_out)[::-1] < lower_bound)
                    index_lower = 0
                    try:
                        index_lower = index_lower_out.index(True)
                    except ValueError:
                        index_lower = self.sim.n_sample_points

                    index_upper_out = list(np.array(self.sim._sim_out)[::-1] > upper_bound)
                    index_upper = 0
                    try:
                        index_upper = index_upper_out.index(True)
                    except ValueError:
                        index_upper = self.sim.n_sample_points
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
            json.dump(extra_info, f, indent=4, default=custom_default)
        print(f"Eval Info: RMSE: {np.mean(rmse)} --- Smoothness: {np.mean(sms)}")

        return extra_info