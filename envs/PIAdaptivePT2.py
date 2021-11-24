import random

import control
import gym
import matplotlib.pyplot as plt
import numpy as np


class PIAdaptivePT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, controller_sample_frequency=100,
                 simulation_time=1.5):
        super(PIAdaptivePT2, self).__init__()
        if oscillating:
            self.sys = control.tf([1], [0.001, 0.005, 1])
        else:
            self.sys = control.tf([1], [0.001, 0.05, 1])

        # init rendering
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

        self.observation_space = gym.spaces.Box(low=np.array([-100, -1, -100, -100]), high=np.array([100, 1, 100, 100]),
                                                shape=(4,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-0.99]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

        # just for logging
        self.actions_log = []
        self.rewards_log = []
        self.observations_log = []
        self.tensorboard_log = {}
        self.dones_log = []

    def reset(self, step_height=None, step_slope=None):
        """
        Important: the observation must be a numpy array; Return after every step with random step between 0, 2
        :return: (np.array)
        """
        if not step_height:
            step_height = random.uniform(-10, 10)
        if not step_slope:
            step_slope = random.uniform(0, 0.5)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(step_slope * self.model_sample_frequency)).tolist()
        # u_step = []
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step

        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0
        self.last_action = 0
        observation = [0, 0, 0, 0]

        return np.array(observation).astype(np.float32)

    def step(self, action):
        controller_p = (action[0] + 1) * 50
        controller_p = np.clip(controller_p, 0, 100)
        # get p and i from action
        controller_i = 0
        # controller_i = 0
        # print(action)

        # close loop with pi controller with given parameters
        pi_controller = control.tf([controller_p, controller_i], [1, 0])
        open_loop = control.series(pi_controller, self.sys)
        closed_loop = control.feedback(open_loop, 1, -1)

        # simulate one controller update step
        start_step = self.simulation_time_steps
        stop_step = self.simulation_time_steps + self.model_steps_per_controller_value
        # try:
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.forced_response(closed_loop,
                                                                 self.t[start_step:stop_step+1],
                                                                 self.u[start_step:stop_step+1],
                                                                 return_x=True)
        else:
            sim_time, out_step, self.last_state = control.forced_response(closed_loop,
                                                                 self.t[start_step:stop_step+1],
                                                                 self.u[start_step:stop_step+1],
                                                                 X0=self.last_state[:, -1],
                                                                 return_x=True)
        self.out = self.out + out_step.tolist()[:-1]
        if stop_step >= len(self.t):
            self.out = self.out + self.out[-1:]
            done = True
        else:
            done = False

        # create observation
        current_set_point = self.u[stop_step-1]
        current_system_output = self.out[stop_step-1]
        current_system_output_dot = (self.out[stop_step-1] - self.out[
            stop_step-1 - self.model_steps_per_controller_value + 1]) / self.model_steps_per_controller_value * 100
        current_system_input = action
        obs = [current_set_point, current_system_input, current_system_output, current_system_output_dot]

        # create reward
        pen_error = np.mean(np.abs(np.array(self.out[start_step:stop_step-1]) - np.array(
            self.u[start_step:stop_step-1])))  # mean error over last simulation step
        pen_error = pen_error * 100
        pen_action = controller_p * 0.1
        pen_integrated = np.square(self.integrated_error) * 0
        offset = 0
        reward = offset - pen_error - pen_action - pen_integrated

        # update simulation variables
        self.simulation_time_steps += self.model_steps_per_controller_value
        # self.last_action = system_input

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        # reward = 1 - reward
        # just for logging
        self.actions_log.append({"reference_value": controller_p,
                                 "action": controller_p})
        self.rewards_log.append({"offset": offset,
                                 "reward": reward,
                                 "pen_error": pen_error,
                                 "pen_action": pen_action,
                                 "pen_integrated": pen_integrated
                                 })
        self.observations_log.append(obs)
        self.dones_log.append(done)
        self.tensorboard_log = {"Obs/Error": obs,
                                "Obs/Integrated_Error": self.integrated_error,
                                "Action/P": controller_p,
                                "Action/I": controller_i,
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
