import json
import pathlib
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from envs.TfSim import TfSim


class DirectControllerPT2(gym.Env):

    def __init__(self, oscillating=True, log=False, reward_function="discrete", observation_function="obs_with_vel",
                 oscillation_pen_gain=0.1,
                 oscillation_pen_fun=np.sqrt, error_pen_fun=None):
        """
        Create a gym environment to directly control the actuating value (u) of a system.
        :param oscillating: If True a
        :param log: Log the simulation outcomes.
        """
        super(DirectControllerPT2, self).__init__()
        if oscillating:
            sys = control.tf([2], [0.001, 0.005, 1])
        else:
            sys = control.tf([2], [0.001, 0.05, 1])

        sys = control.tf([3.55e3], [0.00003, 0.0014, 1])  #  pt2 of dms

        # create simulation object with an arbitrary tf.
        self.sim = TfSim(sys, 10_000, 200, 100, action_scale=500, obs_scale=3_000_000, simulation_time=1.5)
        self.sys_gain = (self.sim.sys.C @ np.linalg.inv(-self.sim.sys.A) @ self.sim.sys.B + self.sim.sys.D)[0][0]

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

        self.oscillation_pen_gain_raising = np.linspace(0, 20, 670)
        self.oscillation_pen_gain = oscillation_pen_gain
        self.oscillation_pen_fun = oscillation_pen_fun
        self.error_pen_fun = error_pen_fun
        # create fifo lists for newest simulation outcomes.
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)

        self.w = []

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
        self.action_space = gym.spaces.Box(low=np.array([-1], dtype=np.float32),
                                           high=np.array([1], dtype=np.float32),
                                           shape=(1,),
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

        # self.oscillation_pen_gain = self.oscillation_pen_gain_raising[self.n_episodes]
        print(self.oscillation_pen_gain)
        print(self.n_episodes)

        # create reference value (w)
        if custom_w is not None:
            assert self.sim.n_sample_points == len(custom_w), "Simulation and input length must not differ"
            self.w = custom_w
        else:
            self.w = self.set_w(step_start, step_end, step_slope)

        # set x0 in state space, to do so compute step response to given w[0]
        T = control.timeresp._default_time_vector(self.sim.sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        U = np.ones_like(T) * self.w[0] * (self.sim.obs_scale/self.sys_gain)
        _, step_response, states = control.forced_response(self.sim.sys, T, U, return_x=True)
        self.sim.last_state = np.array([states[:, -1]]).T

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
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["action"]["value"] = []
            self.episode_log["action"]["change"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None

        # create start observation of new episode
        self.last_system_inputs = deque([(self.w[0] * self.sim.obs_scale)/(self.sys_gain * self.sim.action_scale)] * 3, maxlen=3)
        self.last_system_outputs = deque([(self.sim.last_state[:, -1] @ self.sim.sys.C.T)[0] / self.sim.obs_scale] * 3, maxlen=3)
        self.last_set_points = deque([self.w[0]] * 3, maxlen=3)
        obs = self.observation_function(first=False)
        return obs

    def set_w(self, step_start=None, step_end=None, step_slope=None):
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
        return w

    def _create_obs_with_vel(self):
        """
        Create observation consisting of: set point (w), system output (y), system input (u) and their derivations.
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        system_inputs = np.array(list(self.last_system_inputs))

        outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1 / self.sim.sensor_steps_per_controller_update
        input_vel = (system_inputs[-3] - system_inputs[-1]) * 1 / self.sim.model_steps_per_controller_update
        set_point_vel = (set_points[-2] - set_points[-1]) * 1 / self.sim.sensor_steps_per_controller_update

        obs = [set_points[-1], system_inputs[-1], system_outputs[-1], set_point_vel, input_vel, outputs_vel]

        if self.log:
            self.episode_log["obs"]["set_point"].append(obs[0])
            self.episode_log["obs"]["system_input"].append(obs[1])
            self.episode_log["obs"]["system_output"].append(obs[2])
            self.episode_log["obs"]["set_point_vel"].append(obs[3])
            self.episode_log["obs"]["input_vel"].append(obs[4])
            self.episode_log["obs"]["outputs_vel"].append(obs[5])

        return np.array(obs)

    def _create_obs_last_states(self):
        """
        Create observation consisting of: last 3 set points (w), last 3 system outputs (y), last 3 system inputs (u).
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """

        obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)

        if self.log:
            self.episode_log["obs"]["last_set_points"].append(list(obs[0:3]))
            self.episode_log["obs"]["last_system_inputs"].append(list(obs[3:6]))
            self.episode_log["obs"]["last_system_outputs"].append(list(obs[6:9]))

        return np.array(obs)

    def _create_obs_errors_with_vel(self):
        """
        Create observation consisting of: system error (e), system input (u) and their derivations.
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        system_inputs = np.array(list(self.last_system_inputs))
        errors = (set_points - system_outputs).tolist()

        error_vel = (errors[-2] - errors[-1]) * 1 / self.sim.sensor_steps_per_controller_update
        input_vel = (system_inputs[-3] - system_inputs[-1]) * 1 / self.sim.model_steps_per_controller_update

        obs = [errors[0], system_inputs[-1], error_vel, input_vel]

        if self.log:
            self.episode_log["obs"]["error"].append(obs[0])
            self.episode_log["obs"]["system_input"].append(obs[1])
            self.episode_log["obs"]["error_vel"].append(obs[2])
            self.episode_log["obs"]["input_vel"].append(obs[3])

        return np.array(obs)

    def _create_obs_last_errors(self):
        """
        Create observation consisting of: last 3 system errors (e), last 3 system inputs (u).
        :param first: True if first call in an episode (normally in reset()).
        :return:
        """
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        errors = (set_points - system_outputs).tolist()

        obs = errors + list(self.last_system_inputs)

        if self.log:
            self.episode_log["obs"]["errors"].append(obs[0:3])
            self.episode_log["obs"]["last_system_inputs"].append(obs[3:6])

        return np.array(obs)

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

        # calculate action change
        action_change = self.last_system_inputs[-(self.sim.sensor_steps_per_controller_update + 1)] \
                        - self.last_system_inputs[-self.sim.sensor_steps_per_controller_update]

        pen_error = np.square(e)
        pen_action = np.square(action_change) * 0.01
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

        # calculate action change
        action_change = self.last_system_inputs[-(self.sim.sensor_steps_per_controller_update + 1)] \
                        - self.last_system_inputs[-self.sim.sensor_steps_per_controller_update]

        abs_error = abs(e)

        if self.error_pen_fun:
            pen_error = self.error_pen_fun(abs(e))
        else:
            pen_error = abs(e)

        if self.oscillation_pen_fun:
            pen_action = self.oscillation_pen_fun(abs(action_change)) * self.oscillation_pen_gain
        else:
            pen_action = abs(action_change) * self.oscillation_pen_gain

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
            self.episode_log["rewards"]["pen_action"].append(-pen_action)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
        return reward

    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return:
        """

        # create static input for every simulation step until next update of u.
        system_input_trajectory = [action[0]] * (self.sim.model_steps_per_controller_update + 1)

        if self.log:
            self.episode_log["action"]["value"].append(action[0] * self.sim.action_scale)
            self.episode_log["action"]["change"].append((action[0] - self.last_system_inputs[-1]) * self.sim.action_scale)

        # simulate system until next update of u.
        self.sim.sim_one_step(u=system_input_trajectory, add_noise=True)

        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time used for more precise simulation results.
        # A sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # For the fifo lists the newest sensor values are used.
        for step in range(self.sim.sensor_steps_per_controller_update, 0, -1):
            self.last_system_inputs.append(self.sim.u_sensor[self.sim.current_sensor_step - step])
            self.last_system_outputs.append(self.sim.sensor_out[self.sim.current_sensor_step - step])
            self.last_set_points.append(
                self.w[self.sim.current_simulation_step - (step - 1) * self.sim.model_steps_per_senor_update - 1])

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

        ax[1][2].set_title("FFT")
        actions_fft, actions_fftfreq, N, sm = self.eval_fft()
        ax[1][2].plot(actions_fftfreq, 2 / N * np.abs(actions_fft[0:N // 2]))
        ax[1][2].text(0.5, 0.9, f"Smoothness: {sm}", transform=ax[1][2].transAxes)
        ax[1][2].grid()

        fig.tight_layout()
        return fig

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
                _, _, _, smoothness = self.eval_fft()
                sms.append(smoothness)
                fig = self.create_eval_plot()
                plt.savefig(f"eval\\{folder_name}\\{i}_{step}_{slope}.png")
                plt.close()
                i += 1
                rmse_episode = np.sqrt(np.square(np.array(self.w) - np.array(self.sim._sim_out)))
                rmse.append(rmse_episode)
        mean_episode_reward = np.sum(rewards) / self.n_episodes
        extra_info["mean_episode_reward"] = mean_episode_reward
        extra_info["rmse"] = np.mean(rmse)
        extra_info["smoothness"] = np.mean(sms)

        with open(f"eval\\{folder_name}\\extra_info.json", 'w+') as f:
            json.dump(extra_info, f)
        print(f"Eval Info: RMSE: {np.mean(rmse)} --- Smoothness: {np.mean(sms)}")
