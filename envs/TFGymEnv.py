import abc
import random
from collections import deque

import control
import gym
import numpy as np


class TFGymEnv(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, sensor_sample_frequency=200,
                 controller_update_frequency=100, simulation_time=1.5, custom_sys=None):
        """
        Abstract class for simulating RL with a system given as a Transfer Function. If no custom_sys is given a non
        oscillating PT2 is used.
        :param oscillating: Use a oscillating PT2?
        :param model_sample_frequency: Model sample frequency
        :param sensor_sample_frequency: "Sensor sample frequency"
        :param controller_update_frequency: Update frequency for controller parameters or system input
        :param simulation_time: Simulation time in seconds
        :param custom_sys: Custom TF sys
        """
        super(TFGymEnv, self).__init__()
        if custom_sys:
            self.sys = custom_sys
        elif oscillating:
            self.sys = control.tf([1], [0.001, 0.005, 1])
        else:
            self.sys = control.tf([1], [0.001, 0.05, 1])
        self.sys = control.tf2ss(self.sys)

        # check for correct input
        if model_sample_frequency % controller_update_frequency != 0:
            raise ValueError("model_sample_frequency must be a multiple of controller_update_frequency")
        if model_sample_frequency % controller_update_frequency != 0:
            raise ValueError("model_sample_frequency must be a multiple of sensor_sample_frequency")

        # set model simulation params
        self.model_sample_frequency = model_sample_frequency
        self.sensor_sample_frequency = sensor_sample_frequency
        self.controller_update_frequency = controller_update_frequency
        self.model_steps_per_controller_update = int(self.model_sample_frequency / self.controller_update_frequency)
        self.model_steps_per_senor_update = int(self.model_sample_frequency / self.sensor_sample_frequency)
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_sample_frequency * simulation_time)
        self.t = np.linspace(0, simulation_time, self.n_sample_points)

        # set per episode variables
        self.last_state = None
        self.current_simulation_step = 0
        self.simulation_out = []
        self.integrated_error = 0
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.episode_log = {}

        self.observation_space = gym.spaces.Box(low=np.array([-100] * 9), high=np.array([100] * 9),
                                                shape=(9,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

        # full training log
        self.all_episode_log = []
        self.current_episode = 0

    def reset_episode_vars(self):
        # reset episode specific variables
        self.last_state = None
        self.current_simulation_step = 0
        self.simulation_out = []
        self.integrated_error = 0
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.episode_log = {}

    def reset(self, step_height=None, step_slope=None):
        """
        Important: the observation must be a numpy array; Return after every step with random step between 0, 2
        :return: (np.array)
        """
        self._create_u(step_height, step_slope)
        if self.current_episode > 0:
            self.all_episode_log.append(self.episode_log)
        self.reset_episode_vars()
        obs = self._create_obs(first=True)

        return np.array(obs).astype(np.float32)

    def _create_u(self, step_height=None, step_slope=None):
        if step_height is None:
            step_height = random.uniform(-10, 10)
        if step_slope is None:
            step_slope = random.uniform(0, 0.5)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(step_slope * self.model_sample_frequency)).tolist()
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step

    def _create_obs(self, first=False):
        if first:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
        else:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
        return obs

    def _create_reward(self, start_step, stop_step):
        pen_error = np.mean(np.abs(np.array(self.out[start_step:stop_step:self.model_steps_per_senor_update]) - np.array(
            self.u[start_step:stop_step:self.model_steps_per_senor_update])))  # mean error over last simulation step
        pen_error = pen_error * 100
        pen_action = np.square(self.last_system_inputs[-2] - self.last_system_inputs[-1])
        pen_integrated = np.square(self.integrated_error) * 10
        reward = -pen_error - pen_action - pen_integrated
        return reward

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    def sim_one_step(self, system_input, start_step, stop_step, add_noise=True):
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                          self.t[start_step:stop_step + 1],
                                                                          system_input,
                                                                          return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[start_step:stop_step + 1],
                                                                              system_input,
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
            except ValueError:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[start_step:stop_step + 1],
                                                                              system_input[:-1],
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
        if add_noise:
            out_step = out_step + np.random.normal(0, 0.01, size=out_step.shape[0])
        self.simulation_out = self.simulation_out + out_step.tolist()[:-1]

    def render(self, mode='console'):
        pass
