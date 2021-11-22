import random
from collections import deque
import matplotlib.pyplot as plt
import control
import gym
import numpy as np


class DirectControllerPT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, controller_sample_frequency=100,
                 simulation_time=1.5, eval_plot_mode=False, logger=None):
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
        self.save_renders = True
        self.eval_plot_mode = eval_plot_mode
        self.logger = logger
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
        self.last_gain = 0
        self.last_errors = deque([0] * 5, maxlen=5)

        self.observation_space = gym.spaces.Box(low=np.array([-20, -20, -20]), high=np.array([20, 20, 20]),
                                                shape=(3,),
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
        step_height = random.uniform(1, 10)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(0.05 * self.model_sample_frequency)).tolist()
        # u_step = []
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step

        # reset episode specific variables
        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0
        self.last_gain = 0
        self.last_errors = deque([0] * 5, maxlen=5)

        # initial observation is zero
        return np.array([0, 0, 0]).astype(np.float32)

    def step(self, action):
        # set controller output
        action = action[0] * 50

        # constant_value until next update
        # next_reference_value = np.clip(self.last_gain + action, -10, 10)
        next_reference_value = np.clip(action, -50, 50)
        reference_value = [next_reference_value] * (self.model_steps_per_controller_value + 1)


        # simulate one controller update step
        i = self.simulation_time_steps
        # try:
        if self.last_state is None:

            sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                          self.t[
                                                                          i:i + self.model_steps_per_controller_value + 1],
                                                                          reference_value,
                                                                          return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[
                                                                              i:i + self.model_steps_per_controller_value + 1],
                                                                              reference_value,
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
            except ValueError:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[
                                                                              i:i + self.model_steps_per_controller_value + 1],
                                                                              reference_value[:-1],
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)

        # append simulation step and update simulation time
        self.out = self.out + out_step.tolist()[:-1]
        self.simulation_time_steps += self.model_steps_per_controller_value  #

        # check if simulation/episode is complete
        if self.simulation_time_steps >= len(self.t):
            self.out = self.out + self.out[-1:]
            done = True
            pen_error = np.mean(np.abs(np.array(self.out) - self.u))  # mean error at every time step
            pen_error = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps - 1]) - np.array(
                self.u[i:self.simulation_time_steps - 1])))  # mean error over last simulation step
            # self.render()
        else:
            done = False
            pen_error = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps - 1]) - np.array(
                self.u[i:self.simulation_time_steps - 1])))  # mean error over last simulation step
            # reward = 1

        # error between input and set point and measured value
        error = self.out[self.simulation_time_steps - 1] - self.u[self.simulation_time_steps - 1]
        self.last_errors.append(error)  # append to list of last errors
        # integrate error in current episode
        self.integrated_error += abs(error) * 1 / self.controller_sample_frequency

        derived_error = (self.last_errors[-2] - self.last_errors[-1]) / self.controller_sample_frequency
        derived_error = derived_error * 100  # for equal scaling of obs
        obs = [-error, self.integrated_error, derived_error]


        current_reference_value = self.u[self.simulation_time_steps-1]
        current_out = self.out[self.simulation_time_steps-1]
        current_out_dot = (self.out[self.simulation_time_steps-1] - self.out[
            self.simulation_time_steps - self.model_steps_per_controller_value]) / self.model_steps_per_controller_value * 100
        obs = [current_reference_value, current_out, current_out_dot]
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # create reward
        pen_error = pen_error * 100
        pen_action = np.square(self.last_gain - next_reference_value - (self.u[self.simulation_time_steps-1]
                                                                        - self.u[self.simulation_time_steps-self.model_steps_per_controller_value-1])) * 1
        pen_integrated = np.square(self.integrated_error) * 0
        offset = 0
        reward = offset - pen_error - pen_action - pen_integrated
        # reward = round(reward, 0)
        # just for logging
        self.last_gain = next_reference_value
        self.actions_log.append({"reference_value": next_reference_value,
                                 "action": action})
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
                                "Obs/Vel": derived_error,
                                "Action": reference_value[0],
                                "Reward": reward,
                                "Done": 1 if done else 0}


        return np.array(obs).astype(np.float32), reward, done, info

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
        else:
            self.line1.set_ydata(self.out)
            self.line2.set_ydata(self.u)
            self.line3.set_offsets(np.array([x_points, y_points]).T)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
