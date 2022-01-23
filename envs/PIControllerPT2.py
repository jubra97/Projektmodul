import json
import pathlib
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from envs.IOSim import IOSim


class PIControllerPT2(gym.Env):

    def __init__(self, oscillating=True, log=False, reward_function="discrete", observation_function="obs_with_vel",
                 oscillation_pen_gain=10, oscillation_pen_fun=np.sqrt, error_pen_fun=None):
        """
        Create a gym environment to directly control the actuating value (u) of a system.
        :param oscillating: If True a
        :param log: Log the simulation outcomes.
        """
        super(PIControllerPT2, self).__init__()
        if oscillating:
            self.open_loop_sys = control.tf([2], [0.001, 0.005, 1])
        else:
            self.open_loop_sys = control.tf([2], [0.001, 0.05, 1])

        # sys = control.tf([3.55e3], [0.00003, 0.0014, 1])  #  pt2 of dms
        self.open_loop_sys = control.tf2ss(self.open_loop_sys)

        # create simulation object with an arbitrary tf.
        self.sim = IOSim(None, 10_000, 200, 100, action_scale=10, obs_scale=1, simulation_time=1.5)
        self.open_loop_gain = (self.open_loop_sys.C @ np.linalg.inv(-self.open_loop_sys.A) @ self.open_loop_sys.B + self.open_loop_sys.D)[0][0]

        if reward_function == "discrete":
            self.reward_function = self._create_reward_discrete
        elif reward_function == "normal":
            self.reward_function = self._create_reward
        else:
            raise ValueError(
                "No corresponding reward function could be found. If you want to add your own reward function add the"
                " function to this if else block.")

        if observation_function == "obs_with_vel":
            self.observation_function = self._create_obs_with_vel
        elif observation_function == "last_states":
            self.observation_function = self._create_obs_last_states
        elif observation_function == "error_with_vel":
            self.observation_function = self._create_obs_errors_with_vel
        elif observation_function == "last_errors":
            self.observation_function = self._create_obs_last_errors
        else:
            raise ValueError(
                "No corresponding observation function could be found. If you want to add your own observation"
                "function add the function to this if else block.")

        self.oscillation_pen_gain = oscillation_pen_gain
        self.oscillation_pen_fun = oscillation_pen_fun
        self.error_pen_fun = error_pen_fun
        # create fifo lists for newest simulation outcomes.
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.last_system_errors = deque([0] * 3, maxlen=3)
        self.last_ps = deque([0] * 3, maxlen=3)
        self.last_is = deque([0] * 3, maxlen=3)

        self.w = []
        self.first_step = False

        # create variables for logging
        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log = log
        self.n_episodes = 0
        self.log_all = []

        # define size of observation and action space
        test_obs = self.reset()
        obs_size = test_obs.shape[0]

        self.observation_space = gym.spaces.Box(low=np.array([-100] * obs_size, dtype=np.float32),
                                                high=np.array([100] * obs_size, dtype=np.float32),
                                                shape=(obs_size,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]*2, dtype=np.float32),
                                           high=np.array([1]*2, dtype=np.float32),
                                           shape=(2,),
                                           dtype=np.float32)

    def reset(self, step_start=None, step_end=None, step_slope=None, custom_w=None):
        """
        Reset the environment. Called before every start of a new episode. If no custom reference value (custom_w) is
        given a ramp (or step) is used.
        :param step_height: Height of step/ramp.
        :param step_slope: Slope of Step. If slope is 0 a step is generated.
        :param custom_w:
        :return:
        """

        # add log of last episode to a list of logs of all episodes
        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1

        # reset simulation
        self.sim.reset()

        # create reference value (w)
        if custom_w is not None:
            assert self.sim.n_sample_points == len(custom_w), "Simulation and input length must not differ"
            self.w = custom_w
        else:
            self.w = self.set_w(step_start, step_end, step_slope)
        self.first_step = True

        # reset stuff that is only valid in the same episode
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_system_errors = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.last_ps = deque([0] * 3, maxlen=3)
        self.last_is = deque([0] * 3, maxlen=3)

        # set x0 in state space, to do so compute step response to given w[0]
        T = control.timeresp._default_time_vector(self.open_loop_sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        # gain = (self.sim.sys.C @ np.linalg.inv(-self.sim.sys.A) @ self.sim.sys.B + self.sim.sys.D)[0][0]
        U = np.ones_like(T) * self.w[0][0] * (self.sim.obs_scale/self.open_loop_gain)
        _, step_response, states = control.forced_response(self.open_loop_sys, T, U, return_x=True)
        self.sim.last_state = np.concatenate((np.array([[0]]), states[:, -1:]))

        if self.log:
            self.episode_log["obs"]["last_set_points"] = []
            self.episode_log["obs"]["last_system_inputs"] = []
            self.episode_log["obs"]["last_system_outputs"] = []
            self.episode_log["obs"]["errors"] = []
            self.episode_log["obs"]["error"] = []
            self.episode_log["obs"]["error_vel"] = []
            self.episode_log["obs"]["set_point"] = []
            self.episode_log["obs"]["system_input"] = []
            self.episode_log["obs"]["system_output"] = []
            self.episode_log["obs"]["set_point_vel"] = []
            self.episode_log["obs"]["input_vel"] = []
            self.episode_log["obs"]["outputs_vel"] = []
            self.episode_log["obs"]["system_error"] = []
            self.episode_log["obs"]["error_vel"] = []
            self.episode_log["obs"]["i"] = []
            self.episode_log["obs"]["p"] = []
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["action"]["value"] = []
            self.episode_log["action"]["change"] = []
            self.episode_log["action"]["p_change"] = []
            self.episode_log["action"]["i_change"] = []
            self.episode_log["action"]["p_value"] = []
            self.episode_log["action"]["i_value"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None

        # create start observation of new episode
        self.last_system_inputs = deque([(self.w[0, 0] * self.sim.obs_scale)/(self.open_loop_gain * self.sim.action_scale)] * 3, maxlen=3)
        self.last_system_outputs = deque([(states[:, -1] @ self.open_loop_sys.C.T)[0] / self.sim.obs_scale] * 3, maxlen=3)
        self.last_set_points = deque([self.w[0, 0]] * 3, maxlen=3)
        self.last_system_errors = deque([0] * 3, maxlen=3)
        obs = self.observation_function()
        return obs

    def set_w(self, step_start=None, step_end=None, step_slope=None, add_noise=True):
        """
        Create reference value (w) as ramp or step
        :param step_height: Height of ramp/ step
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

        if add_noise:
            noise = np.random.normal(0, 0.001, self.sim.n_sample_points)
        else:
            noise = [0] * self.sim.n_sample_points

        sys_input = np.array([w, noise])
        return sys_input

    def _create_obs_with_vel(self):
        """
        Create observation consisting of: set point (w), system output (y), system input (u) and their derivations.
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        system_errors = np.array(list(self.last_system_errors))
        p_s = np.array(list(self.last_ps))
        i_s = np.array(list(self.last_is))

        outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1 / self.sim.sensor_steps_per_controller_update
        set_point_vel = (set_points[-2] - set_points[-1]) * 1 / self.sim.sensor_steps_per_controller_update
        error_vel = (system_errors[-2] - system_errors[-1]) * 1 / self.sim.sensor_steps_per_controller_update

        # obs = [p_s[-1], i_s[-1], set_points[-1], system_outputs[-1], system_errors[-1], set_point_vel, outputs_vel, error_vel]

        obs = [p_s[-1], i_s[-1], system_errors[-1], error_vel]

        # if self.log:
        #     self.episode_log["obs"]["p"].append(obs[0])
        #     self.episode_log["obs"]["i"].append(obs[1])
        #     self.episode_log["obs"]["set_point"].append(obs[2])
        #     self.episode_log["obs"]["system_output"].append(obs[3])
        #     self.episode_log["obs"]["system_error"].append(obs[4])
        #     self.episode_log["obs"]["set_point_vel"].append(obs[5])
        #     self.episode_log["obs"]["outputs_vel"].append(obs[6])
        #     self.episode_log["obs"]["error_vel"].append(obs[7])

        if self.log:
            self.episode_log["obs"]["p"].append(obs[0])
            self.episode_log["obs"]["i"].append(obs[1])
            self.episode_log["obs"]["system_error"].append(obs[2])
            self.episode_log["obs"]["error_vel"].append(obs[3])


        return np.array(obs)

    # def _create_obs_last_states(self, first=False):
    #     """
    #     Create observation consisting of: last 3 set points (w), last 3 system outputs (y), last 3 system inputs (u).
    #     :param first: True if first call in an episode (normally in reset()).
    #     :return:
    #     """
    #     if first:
    #         #  @ is matrix/vector multiply
    #         obs = [0, 0, self.w[0]] + list(self.last_system_inputs) +\
    #               [0, 0, (self.sim.last_state[:, -1] @ self.sim.sys.C.T)[0] / self.sim.obs_scale]
    #     else:
    #         obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
    #
    #     if self.log:
    #         self.episode_log["obs"]["last_set_points"].append(list(obs[0:3]))
    #         self.episode_log["obs"]["last_system_inputs"].append(list(obs[3:6]))
    #         self.episode_log["obs"]["last_system_outputs"].append(list(obs[6:9]))
    #     return np.array(obs)
    #
    # def _create_obs_errors_with_vel(self, first=False):
    #     """
    #     Create observation consisting of: system error (e), system input (u) and their derivations.
    #     :param first: True if first call in an episode (normally in reset()).
    #     :return:
    #     """
    #     set_points = np.array(list(self.last_set_points))
    #     system_outputs = np.array(list(self.last_system_outputs))
    #     system_inputs = np.array(list(self.last_system_inputs))
    #     errors = (set_points - system_outputs).tolist()
    #
    #     error_vel = (errors[-2] - errors[-1]) * 1 / self.sim.sensor_steps_per_controller_update
    #     input_vel = (system_inputs[-3] - system_inputs[-1]) * 1 / self.sim.model_steps_per_controller_update
    #
    #     if first:
    #         error = self.w[0] - system_outputs[-1]
    #         obs = [error, system_inputs[-1], error_vel, input_vel]
    #     else:
    #         obs = [errors[0], system_inputs[-1], error_vel, input_vel]
    #
    #     if self.log:
    #         self.episode_log["obs"]["error"].append(obs[0])
    #         self.episode_log["obs"]["system_input"].append(obs[1])
    #         self.episode_log["obs"]["error_vel"].append(obs[2])
    #         self.episode_log["obs"]["input_vel"].append(obs[3])
    #     return np.array(obs)
    #
    # def _create_obs_last_errors(self, first=False):
    #     """
    #     Create observation consisting of: last 3 system errors (e), last 3 system inputs (u).
    #     :param first: True if first call in an episode (normally in reset()).
    #     :return:
    #     """
    #     set_points = np.array(list(self.last_set_points))
    #     system_outputs = np.array(list(self.last_system_outputs))
    #     errors = (set_points - system_outputs).tolist()
    #
    #     if first:
    #         obs = errors + list(self.last_system_inputs)
    #     else:
    #         obs = errors + list(self.last_system_inputs)
    #
    #     if self.log:
    #         self.episode_log["obs"]["errors"].append(obs[0:3])
    #         self.episode_log["obs"]["last_system_inputs"].append(obs[3:6])
    #     return np.array(obs)

    def _create_reward(self):
        """
        Create reward as a combination of the current error e between y and w and the change of of the action (to
        smaller the oscillation in u).
        :return: reward
        """
        # get latest system attributes and calculate error/ integrated error
        y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
        w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
        e = np.mean(w - y)
        u = np.array(list(self.last_system_inputs)[-2:])
        u_change = abs(u[-1] - u[-2])

        if e > 1_000_000:
            print(f"E: {e}")
            print(f"P: {self.p}")
            print(f"I: {self.i}")

        # calculate action change
        action_change = self.last_system_inputs[-(self.sim.sensor_steps_per_controller_update + 1)] \
                        - self.last_system_inputs[-self.sim.sensor_steps_per_controller_update]

        pen_error = np.abs(e)
        pen_action = np.sqrt(u_change) * 1
        # pen_integrated = np.square(self.integrated_error) * 0

        reward = 0
        reward -= pen_error
        reward -= pen_action

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)

        return reward

    def _create_reward_discrete(self):
        # get latest system attributes and calculate error/ integrated error
        y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
        w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
        e = np.mean(w - y)
        u = np.array(list(self.last_system_inputs)[-2:])
        u_change = abs(u[-1] - u[-2])

        # calculate action change
        # action_change = self.last_system_inputs[-(self.sim.sensor_steps_per_controller_update + 1)] \
        #                 - self.last_system_inputs[-self.sim.sensor_steps_per_controller_update]

        abs_error = abs(e)

        if self.error_pen_fun:
            pen_error = self.error_pen_fun(abs(e))
        else:
            pen_error = abs(e)

        if self.oscillation_pen_fun:
            pen_u_change = self.oscillation_pen_fun(abs(u_change)) * self.oscillation_pen_gain
        else:
            pen_u_change = abs(u_change) * self.oscillation_pen_gain

        reward = 0
        if abs_error < 0.5:
            reward += 1
        if abs_error < 0.1:
            reward += 2
        if abs_error < 0.05:
            reward += 3
        if abs_error < 0.02:
            reward += 5
        if abs_error < 0.005:
            reward += 10

        reward -= pen_error
        reward -= pen_u_change

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_action"].append(-pen_u_change)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
        return reward

    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return:
        """

        # system is not defined if p and i is zero
        smallest_possible_pos_value = np.nextafter(0, 1)
        controller_p = np.clip(action[0] + 1, 1e-3, 2)
        controller_i = np.clip(action[1] + 1, 1e-3, 2)

        # create static input for every simulation step until next update of u.
        system_input_trajectory = [action[0]] * (self.sim.model_steps_per_controller_update + 1)

        if self.log:
            self.episode_log["action"]["p_value"].append(controller_p)
            self.episode_log["action"]["p_change"].append(controller_p - self.last_ps[-1])
            self.episode_log["action"]["i_value"].append(controller_i)
            self.episode_log["action"]["i_change"].append(controller_i - self.last_is[-1])

        # update system with new p and i; simulate next time step
        self.sim.sys = self.create_io_sys(controller_p, controller_i)
        self.sim.sim_one_step(w=self.w[:, self.sim.current_simulation_step:self.sim.current_simulation_step+self.sim.model_steps_per_controller_update+1])

        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time used for more precise simulation results.
        # A sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # For the fifo lists the newest sensor values are used.
        for step in range(self.sim.sensor_steps_per_controller_update, 0, -1):
            self.last_ps.append(controller_p)
            self.last_is.append(controller_i)
            self.last_system_errors.append(self.sim.sensor_out[3, self.sim.current_sensor_step - step])  # e
            self.last_system_inputs.append(self.sim.sensor_out[2, self.sim.current_sensor_step - step])  # u
            self.last_system_outputs.append(self.sim.sensor_out[1, self.sim.current_sensor_step - step])  # y
            self.last_set_points.append(self.w[0, self.sim.current_simulation_step - (step - 1) * self.sim.model_steps_per_senor_update - 1])  # w

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
        self.first_step = False
        return obs, reward, done, {}

    def create_io_sys(self, p, i):
        p = p * self.sim.action_scale
        i = i * self.sim.action_scale
        self.p = p
        self.i = i

        pi_controller = control.tf2ss(control.tf([p, i], [1, 0]))
        if self.first_step:
            self.sim.last_state[0, 0] = self.w[0, 0] / (i * self.open_loop_gain)

        io_open_loop = control.LinearIOSystem(self.open_loop_sys, inputs="u", outputs="y", name="open_loop")
        io_pi = control.LinearIOSystem(pi_controller, inputs="e", outputs="u", name="controller")
        w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
        y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

        closed_loop = control.interconnect([w_y_comp, io_pi, io_open_loop, y_noise], name="closed_loop",
                                           inplist=["w", "noise"],
                                           outlist=["y", "y_noisy", "u", "e"])
        return closed_loop

    def render(self, mode="human"):
        """
        Must be here to implement abstract methods of super class.
        :param mode:
        :return:
        """
        ...

    def eval_fft(self):
        """
        Calculate fft of the action signal for one episode. Also compute the action smoothness value in
        https: // arxiv.org / pdf / 2012.06644.pdf.
        :return:
        """
        N = 150
        T = 1 / 100

        actions = self.episode_log["action"]["value"]
        actions = actions - np.mean(actions)
        actions_fft = fft(actions)
        actions_fftfreq = fftfreq(N, T)[:N // 2]

        # https: // arxiv.org / pdf / 2012.06644.pdf Smoothness Measurement
        sm = (2 / actions_fftfreq.shape[0]) * np.sum(actions_fftfreq * 2 / N * np.abs(actions_fft[0:N // 2]))
        return actions_fft, actions_fftfreq, N, sm

    def create_eval_plot(self):
        """
        Crate a plot for tensorboard while training and afterwards for evaluation.
        :return:
        """
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        timestamps = np.linspace(0, self.sim.simulation_time, int(self.sim.simulation_time * self.sim.controller_freq))

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

        # ax[1][2].set_title("FFT")
        # actions_fft, actions_fftfreq, N, sm = self.eval_fft()
        # ax[1][2].plot(actions_fftfreq, 2 / N * np.abs(actions_fft[0:N // 2]))
        # ax[1][2].text(0.5, 0.9, f"Smoothness: {sm}", transform=ax[1][2].transAxes)
        # ax[1][2].grid()

        fig.tight_layout()
        return fig

    def eval(self, model, folder_name):
        """
        Run an evaluation with different steps and ramps. Create a plot for every run and save it. Also save a json file
        with some statistics of a run.
        :param model: Model to be used for action prediction.
        :param folder_name: Folder to save evaluation in.
        """
        steps = np.linspace(-1, 1, 20)
        slopes = np.linspace(0, 0.5, 3)
        i = 1
        pathlib.Path(f"eval\\{folder_name}").mkdir(exist_ok=True)
        rewards = []
        rmse = []
        sms = []
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
                # _, _, _, smoothness = self.eval_fft()
                # sms.append(smoothness)
                fig = self.create_eval_plot()
                plt.savefig(f"eval\\{folder_name}\\{i}_{step}_{slope}.png")
                plt.close()
                i += 1
                rmse_episode = np.sqrt(np.square(np.array(self.w) - np.array(self.sim._sim_out)))
                rmse.append(rmse_episode)
        mean_episode_reward = np.sum(rewards) / self.n_episodes
        extra_info["mean_episode_reward"] = mean_episode_reward
        extra_info["rmse"] = np.mean(rmse)
        # extra_info["smoothness"] = np.mean(sms)

        with open(f"eval\\{folder_name}\\extra_info.json", 'w+') as f:
            json.dump(extra_info, f)
        print(f"Eval Info: RMSE: {np.mean(rmse)} --- Smoothness: {np.mean(sms)}")
