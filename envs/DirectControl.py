import abc
import itertools
from collections import deque

import gym

import numpy as np
from scipy.fft import fft, fftfreq


class DirectController(gym.Env, abc.ABC):

    def __init__(self,
                 log=False,
                 output_freq=100,
                 sensor_freq=4000,
                 reward_kwargs=None,
                 observation_kwargs=None,
                 ):
        """
        Create a gym environment that is designed to directly control the actuating value (u) of a system. It is a
        abstract class that hols the reward and observation functions for the simulation and online implementations. It
        also hols variables for logging.
        :param log: Log the simulation outcomes.
        :param output_freq: Frequency for u update.
        :param sensor_freq: Frequency of new sensor update data.
        :param reward_kwargs: Dict with extra options for the reward function
        :param observation_kwargs: Dict with extra option for the observation function
        """
        super(DirectController, self).__init__()

        self.reward_kwargs = reward_kwargs if reward_kwargs else {}
        self.observation_kwargs = observation_kwargs if observation_kwargs else {}

        # set reward function
        reward_function = self.reward_kwargs.get("function", "normal")
        if reward_function == "normal":
            self.reward_function = self._create_reward
        elif reward_function == "custom":
            print("add your own reward function")
        else:
            raise ValueError(
                "No corresponding reward function could be found. If you want to add your own reward function add the"
                " function to this if else block.")

        # set observation function
        observation_function = self.observation_kwargs.get("function", "error_with_vel")
        if observation_function == "raw_with_vel":
            self.observation_function = self._obs_raw_with_vel
        elif observation_function == "raw_with_last_states":
            self.observation_function = self._obs_raw_with_last_states
        elif observation_function == "error_with_vel":
            self.observation_function = self._obs_errors_with_vel
        elif observation_function == "error_with_last_states":
            self.observation_function = self._obs_errors_with_last_states
        elif observation_function == "error_with_extra_components":
            self.observation_function = self._obs_error_with_custom_extra_data
        else:
            raise ValueError(
                "No corresponding observation function could be found. If you want to add your own observation"
                "function add the function to this if else block.")

        self.sensor_freq = sensor_freq  # Hz
        self.output_freq = output_freq  # Hz
        assert sensor_freq % output_freq == 0, "sensor_freq must be a multiple ot output_freq!"
        self.measurements_per_output_update = int(sensor_freq / output_freq)

        # create fifo lists for latest measurement points, new data is inserted from the right side. Because of this the
        # most recent system output value is in self.last_y[-1].
        self.nbr_measurements_to_keep = self.measurements_per_output_update * 2
        self.last_u = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_y = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

        self.integrated_error = 0

        # create variables for logging
        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log = log
        self.n_episodes = 0
        self.log_all_episodes = []

        # define size of observation space
        test_obs = self.reset()
        obs_size = test_obs.shape[0]
        self.observation_space = gym.spaces.Box(low=np.array([-100] * obs_size, dtype=np.float32),
                                                high=np.array([100] * obs_size, dtype=np.float32),
                                                shape=(obs_size,),
                                                dtype=np.float32)

        # action space size is always one as u is calculated directly
        self.action_space = gym.spaces.Box(low=np.array([-1], dtype=np.float32),
                                           high=np.array([1], dtype=np.float32),
                                           shape=(1,),
                                           dtype=np.float32)

    def reset(self, *args, **kwargs):
        """
        Reset the environment. Called before every start of a new episode. Calls custom_reset that must be implemented
        in a concrete implementation of this class. Resets logging for one episode.
        :return:
        """

        # add log of last episode to a list of logs of all episodes
        if self.n_episodes != 0:
            self.log_all_episodes.append(self.episode_log)
        self.n_episodes += 1

        if self.log:
            self.episode_log["obs"]["last_set_points"] = []
            self.episode_log["obs"]["last_system_inputs"] = []
            self.episode_log["obs"]["last_system_outputs"] = []
            self.episode_log["obs"]["last_errors"] = []
            self.episode_log["obs"]["error"] = []
            self.episode_log["obs"]["error_vel"] = []
            self.episode_log["obs"]["set_point"] = []
            self.episode_log["obs"]["system_input"] = []
            self.episode_log["obs"]["system_output"] = []
            self.episode_log["obs"]["set_point_vel"] = []
            self.episode_log["obs"]["input_vel"] = []
            self.episode_log["obs"]["output_vel"] = []
            self.episode_log["obs"]["error_integrated"] = []
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["action"]["value"] = []
            self.episode_log["action"]["change"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None

        self.integrated_error = 0
        self.custom_reset(*args, **kwargs)
        obs = self.observation_function()
        return obs

    @abc.abstractmethod
    def custom_reset(self, *args, **kwargs):
        """
        Reset env.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def _obs_raw_with_vel(self):
        """
        Create observation consisting of: set point (w), system output (y), system input (u) and their derivations.
        Add a "average_length" field to observation_kwargs to generate a observation with averaged sensor data.
        :return: Observation
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        system_inputs = np.array(list(self.last_u))
        average_length = self.observation_kwargs.get("average_length", 1)

        set_point = np.mean(set_points[-average_length:])
        last_set_point = np.mean(set_points[-average_length * 2:-average_length])
        set_point_vel = (set_point - last_set_point) * (average_length / self.sensor_freq)

        system_output = np.mean(system_outputs[-average_length:])
        last_system_output = np.mean(system_outputs[-average_length * 2:-average_length])
        system_output_vel = (system_output - last_system_output) * (average_length / self.sensor_freq)

        system_input = np.mean(system_inputs[-average_length:])
        # ensure that a change in u was made to calculate input velocity
        if average_length < self.measurements_per_output_update:
            system_input_vel = (system_inputs[-(self.measurements_per_output_update + 1)] - system_inputs[-1]) * 1 / self.measurements_per_output_update
        else:
            last_system_input = np.mean(system_inputs[-average_length * 2:-average_length])
            system_input_vel = (system_input - last_system_input) * (average_length / self.sensor_freq)

        obs = [set_point, system_input, system_output, set_point_vel, system_input_vel, system_output_vel]

        if self.log:
            self.episode_log["obs"]["set_point"].append(obs[0])
            self.episode_log["obs"]["system_input"].append(obs[1])
            self.episode_log["obs"]["system_output"].append(obs[2])
            self.episode_log["obs"]["set_point_vel"].append(obs[3])
            self.episode_log["obs"]["input_vel"].append(obs[4])
            self.episode_log["obs"]["output_vel"].append(obs[5])

        return np.array(obs)

    def _obs_raw_with_last_states(self):
        """
        Create observation consisting of: last set points (w), last system outputs (y), last system inputs (u).
        Add a "history_length" filed to observation_kwargs to determine how many of last states should be added.
        :return: Observation
        """
        history_length = self.observation_kwargs.get("history_length", 3)
        use_u = self.observation_kwargs.get("use_u", True)

        # deque does not support slicing so a little workaround is needed, get newest "history_length" states
        w_history = list(itertools.islice(self.last_w, len(self.last_w) - history_length, len(self.last_w)))
        u_history = list(itertools.islice(self.last_u, len(self.last_u) - history_length, len(self.last_u)))
        y_history = list(itertools.islice(self.last_y, len(self.last_y) - history_length, len(self.last_y)))

        if use_u:
            obs = w_history + u_history + y_history
        else:
            obs = w_history + y_history

        if self.log:
            self.episode_log["obs"]["last_set_points"].append(list(obs[0:history_length]))
            if use_u:
                self.episode_log["obs"]["last_system_inputs"].append(list(obs[history_length:history_length * 2]))
                self.episode_log["obs"]["last_system_outputs"].append(list(obs[history_length * 2:history_length * 3]))
            else:
                self.episode_log["obs"]["last_system_outputs"].append(list(obs[history_length:history_length * 2]))
        return np.array(obs)

    def _obs_errors_with_vel(self):
        """
        Create observation consisting of: system error (e), derivation of error (e_dot) and system input (u_dot).
        :return:
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        system_inputs = np.array(list(self.last_u))
        errors = set_points - system_outputs
        average_length = self.observation_kwargs.get("average_length", 1)

        error = np.mean(errors[-average_length:])
        error_last = np.mean(errors[-average_length * 2:-average_length])
        error_vel = (error - error_last) * (average_length / self.sensor_freq)

        # ensure that a change in u was made to calculate input velocity
        if average_length < self.measurements_per_output_update:
            system_input_vel = (system_inputs[-(self.measurements_per_output_update + 1)] - system_inputs[-1])\
                               * 1 / self.measurements_per_output_update
        else:
            system_input = np.mean(system_inputs[-average_length:])
            system_input_last = np.mean(system_inputs[-average_length * 2:-average_length])
            system_input_vel = (system_input - system_input_last) * (average_length / self.sensor_freq)

        obs = [error, error_vel, system_input_vel]

        if self.log:
            self.episode_log["obs"]["error"].append(obs[0])
            self.episode_log["obs"]["error_vel"].append(obs[1])
            self.episode_log["obs"]["input_vel"].append(obs[2])
        return np.array(obs)

    def _obs_errors_with_last_states(self):
        """
        Create observation consisting of: last system errors (e), last system inputs (u).
        Add a "history_length" filed to observation_kwargs to determine how many of last states should be added.
        :return: Observation
        """
        history_length = self.observation_kwargs.get("history_length", 3)

        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        errors = (set_points - system_outputs).tolist()

        obs = errors[-history_length:]

        if self.log:
            self.episode_log["obs"]["last_errors"].append(obs[0:history_length])

        return np.array(obs)

    def _obs_error_with_custom_extra_data(self):
        """
        Create observation consisting of: system error (e) and configurable extra observations. Define in a dict named
        "obs_config" in observation_kwargs. Mainly used for testing.
        :return: Observation
        """
        set_points = np.array(list(self.last_w))
        system_outputs = np.array(list(self.last_y))
        system_inputs = np.array(list(self.last_u))
        errors = set_points - system_outputs
        average_length = self.observation_kwargs.get("average_length", 1)

        error = np.mean(errors[-average_length:])
        error_last = np.mean(errors[-average_length * 2:-average_length])
        error_vel = (error - error_last) * (average_length / self.sensor_freq)
        self.integrated_error = error * 1 / self.output_freq

        system_input = np.mean(system_inputs[-average_length:])
        # ensure that a change in u was made to calculate input velocity
        if average_length < self.measurements_per_output_update:
            system_input_vel = (system_inputs[-(self.measurements_per_output_update + 1)] - system_inputs[-1])\
                               * 1 / self.measurements_per_output_update
        else:
            system_input = np.mean(system_inputs[-average_length:])
            system_input_last = np.mean(system_inputs[-average_length * 2:-average_length])
            system_input_vel = (system_input - system_input_last) * (average_length / self.sensor_freq)

        system_output = np.mean(system_outputs[-average_length:])
        system_output_last = np.mean(system_outputs[-average_length * 2:-average_length])
        system_output_vel = (system_output - system_output_last) * (average_length / self.sensor_freq)

        obs_config = self.observation_kwargs.get("obs_config", {})
        obs = [error]
        if self.log:
            self.episode_log["obs"]["error"].append(obs[-1])
        if "i" in obs_config and obs_config["i"]:
            obs.append(self.integrated_error)
            if self.log:
                self.episode_log["obs"]["error_integrated"].append(obs[-1])
        if "d" in obs_config and obs_config["d"]:
            obs.append(error_vel)
            if self.log:
                self.episode_log["obs"]["error_vel"].append(obs[-1])
        if "input_vel" in obs_config and obs_config["input_vel"]:
            obs.append(system_input_vel)
            if self.log:
                self.episode_log["obs"]["input_vel"].append(obs[-1])
        if "output_vel" in obs_config and obs_config["output_vel"]:
            obs.append(system_output_vel)
            if self.log:
                self.episode_log["obs"]["output_vel"].append(obs[-1])
        if "output" in obs_config and obs_config["output"]:
            obs.append(system_output)
            if self.log:
                self.episode_log["obs"]["system_output"].append(obs[-1])
        if "input" in obs_config and obs_config["input"]:
            obs.append(system_input)
            if self.log:
                self.episode_log["obs"]["system_input"].append(obs[-1])

        return np.array(obs)

    def _create_reward(self):
        """
        Create reward as a combination of the current error e between y and w and the change of of the action (to reach
        smaller the oscillation in u).
        The Reward function is customizable with the different keyword in the reward_kwargs:
        discrete_bonus: Add a discrete bonus if e is small, helps to achieve more precise final value
        error_pen_fun: Use a function to calculate the penalty generated by e. E.g np.sqrt, or np.square
        oscillation_pen_fun: Use a function to calculate the penalty generated by input change. E.g np.sqrt, or np.square
        oscillation_pen_gain: Weight between error_pen (1) and input_oscillation_pen (weight)
        oscillation_pen_dependent_on_error: weight input_oscillation_pen with 1/e; by this oscillation is higher
                                            penalized if the error is small
        :return: reward
        """
        error_pen_fun = self.reward_kwargs.get("error_pen_fun", None)
        oscillation_pen_fun = self.reward_kwargs.get("oscillation_pen_fun", np.sqrt)
        oscillation_pen_gain = self.reward_kwargs.get("oscillation_pen_gain", 1)
        add_discrete_bonus = self.reward_kwargs.get("discrete_bonus", True)
        oscillation_pen_dependent_on_error = self.reward_kwargs.get("oscillation_pen_dependent_on_error", False)

        # calculate mean error since last output update step
        y = np.array(list(self.last_y)[-self.measurements_per_output_update:])
        w = np.array(list(self.last_w)[-self.measurements_per_output_update:])
        e = np.mean(w - y)

        # calculate difference between last action change
        action_change = (self.last_u[-(self.measurements_per_output_update + 1)]
                         - self.last_u[-self.measurements_per_output_update]) \
                        * (1 / self.measurements_per_output_update)
        # action_change = y[-1] - y[-2]

        abs_error = abs(e)
        abs_action_change = abs(action_change)

        if error_pen_fun:
            pen_error = error_pen_fun(abs_error)
        else:
            pen_error = abs_error

        if oscillation_pen_dependent_on_error:
            abs_action_change = (1 / abs_error) * abs_action_change
        if oscillation_pen_fun:
            pen_action = oscillation_pen_fun(abs_action_change) * oscillation_pen_gain
        else:
            pen_action = abs_action_change * oscillation_pen_gain

        reward = 0

        if add_discrete_bonus:
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
            if abs_error < 0.0005:
                reward += 10

        reward -= pen_error
        reward -= pen_action

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)

        return reward

    @abc.abstractmethod
    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return: observation, reward, done, info
        """
        raise NotImplementedError()

    def render(self, mode="human"):
        """
        Must be here to implement abstract methods of super class.
        :param mode:
        :return:
        """
        ...

    def eval_fft(self):
        """
        Calculate fft of the action signal for one episode. Also compute the action smoothness value as shown in
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


