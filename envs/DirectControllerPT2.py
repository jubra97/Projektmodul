import random
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np


class DirectControllerPT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, controller_sample_frequency=100,
                 simulation_time=1.5):
        super(DirectControllerPT2, self).__init__()
        if oscillating:
            self.sys = control.tf([1], [0.001, 0.005, 1])
        else:
            self.sys = control.tf([1], [0.001, 0.05, 1])

        # # init rendering
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.first_flag = True

        # set model simulation params
        self.model_sample_frequency = model_sample_frequency
        self.controller_sample_frequency = controller_sample_frequency
        self.model_steps_per_controller_value = int(self.model_sample_frequency / self.controller_sample_frequency)
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_sample_frequency * simulation_time)
        self.t = np.linspace(0, simulation_time, self.n_sample_points)
        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0
        self.last_action = 0
        self.last_errors = deque([0] * 5, maxlen=5)

        self.observation_space = gym.spaces.Box(low=np.array([-100, -1, -100, -100]), high=np.array([100, 1, 100, 100]),
                                                shape=(4,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

        # just for logging
        self.actions_log = []
        self.rewards_log = []
        self.observations_log = []
        self.tensorboard_log = {}
        self.dones_log = []

    def reset(self):
        """
        Important: the observation must be a numpy array; Return after every step with random step between 0, 2
        :return: (np.array)
        """

        # create u (e.g. step)
        step_height = random.uniform(-10, 10)
        step_time = random.uniform(0, 0.5)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(step_time * self.model_sample_frequency)).tolist()
        # u_step = []
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step

        # reset episode specific variables
        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0
        self.last_action = 0
        self.last_errors = deque([0] * 5, maxlen=5)

        # initial observation is zero
        return np.array([0, 0, 0, 0]).astype(np.float32)

    def step(self, action):
        action = action[0]
        # constant_value until next update
        # next_reference_value = np.clip(self.last_gain + action, -10, 10)
        system_input = np.clip(action * 50, -50, 50)
        system_input_trajectory = [system_input] * (self.model_steps_per_controller_value + 1)

        # simulate one controller update step
        start_step = self.simulation_time_steps
        stop_step = self.simulation_time_steps + self.model_steps_per_controller_value - 1
        self.sim_one_step(system_input_trajectory, start_step, stop_step+1)

        # check if simulation/episode is complete
        if (stop_step + 1) >= len(self.t):
            self.out = self.out + self.out[-1:]  # add last value to match sim time
            done = True
        else:
            done = False

        # create observation with errors
        # error between system output and set point
        # error = self.out[stop_step] - self.u[stop_step]
        # self.last_errors.append(error)  # append to list of last errors
        # # integrate error in current episode TODO: Use absolute or non absoulte integrated eroor
        # self.integrated_error += abs(error) * 1 / self.controller_sample_frequency
        # derived_error = (self.last_errors[-2] - self.last_errors[-1]) / self.controller_sample_frequency
        # derived_error = derived_error * 100  # for equal scaling of obs
        # obs = [-error, self.integrated_error, derived_error]

        # create observation with system and input state
        current_set_point = self.u[stop_step]
        current_system_output = self.out[stop_step]
        current_system_output_dot = (self.out[stop_step] - self.out[
            stop_step - self.model_steps_per_controller_value+1]) / self.model_steps_per_controller_value * 100
        current_system_input = action
        obs = [current_set_point, current_system_input, current_system_output, current_system_output_dot]
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # create reward
        pen_error = np.mean(np.abs(np.array(self.out[start_step:stop_step]) - np.array(
            self.u[start_step:stop_step])))  # mean error over last simulation step
        pen_error = pen_error * 100
        pen_action = np.square(self.last_action - system_input)
        pen_integrated = np.square(self.integrated_error) * 0
        offset = 0
        reward = offset - pen_error - pen_action - pen_integrated

        # update simulation variables
        self.simulation_time_steps += self.model_steps_per_controller_value
        self.last_action = system_input

        # just for logging
        self.actions_log.append({"reference_value": system_input,
                                 "action": system_input})
        self.rewards_log.append({"offset": offset,
                                 "reward": reward,
                                 "pen_error": pen_error,
                                 "pen_action": pen_action,
                                 "pen_integrated": pen_integrated
                                 })
        self.observations_log.append(obs)
        self.dones_log.append(done)
        self.tensorboard_log = {"Obs/RefVal": obs[0],
                                "Obs/CurrentOut": obs[1],
                                "Obs/CurrentOutDot": obs[2],
                                "Action": system_input,
                                "Reward": reward,
                                "Done": 1 if done else 0}

        return np.array(obs).astype(np.float32), reward, done, info

    def sim_one_step(self, system_input, start_step, stop_step):
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
        self.out = self.out + out_step.tolist()[:-1]

    def render(self, mode='console'):
        x_points = [x for x in range(0, len(self.t), self.model_steps_per_controller_value)]
        y_points = np.array(self.out)[x_points]
        x_points = np.array(x_points) / self.model_sample_frequency
        if self.first_flag:
            self.ax.set_ylim(-0.1, 11)

            self.line1, = self.ax.plot(self.t, self.out, "r-")
            self.line2, = self.ax.plot(self.t, self.u)
            self.line3 = self.ax.scatter(x_points, y_points)
            self.ax.grid()
            self.first_flag = False
            plt.show()
        else:
            self.line1.set_ydata(self.out)
            self.line2.set_ydata(self.u)
            self.line3.set_offsets(np.array([x_points, y_points]).T)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
