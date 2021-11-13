import random

import control
import gym
import matplotlib.pyplot as plt
import numpy as np


class StepAdaptivePT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, simulation_time=1.5):
        super(StepAdaptivePT2, self).__init__()
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
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_sample_frequency * simulation_time)
        self.t = np.linspace(0, simulation_time, self.n_sample_points)

        self.observation_space = gym.spaces.Box(low=-1, high=11, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), shape=(2,),
                                           dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array; Return after every step with random step between 0, 2
        :return: (np.array)
        """
        step_height = random.uniform(1, 1)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step))
        self.u = u_before_step + u_after_step
        return np.array([step_height]).astype(np.float32)

    def step(self, action):
        # get p and i from action
        # action = (action + 1) * 500
        p = action[0]
        i = action[1]
        # print(action)

        # close loop with pi controller with given parameters
        pi_controller = control.tf([p, i], [1, 0])
        open_loop = control.series(pi_controller, self.sys)
        closed_loop = control.feedback(open_loop, 1, -1)

        # simulate step
        try:
            self.out = control.forced_response(closed_loop, self.t, self.u)
            # error between step and simulation result; TODO: Sum or mean?; Normalize with step height?
            reward = -(np.square(self.out[1] - self.u)).sum() / self.u[-1]
        except ValueError:
            reward = -np.inf
        # self.render()
        # Always done
        done = True

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([-10]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if self.first_flag:
            self.ax.set_ylim(-0.1, 11)
            self.line1, = self.ax.plot(self.out[0], self.out[1], "r-")
            self.line2, = self.ax.plot(self.out[0], self.u)
            self.ax.grid()
            self.first_flag = False
        else:
            self.line1.set_ydata(self.out[1])
            self.line2.set_ydata(self.u)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


class FullAdaptivePT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, controller_sample_frequency=100,
                 simulation_time=1.5):
        super(FullAdaptivePT2, self).__init__()
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

        self.observation_space = gym.spaces.Box(low=np.array([-1000, -1000]), high=np.array([1000, 1000]), shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-0.99, -0.99]), high=np.array([1, 1]), shape=(2,),
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
        step_height = random.uniform(4.99, 5)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(0.05 * self.model_sample_frequency)).tolist()
        # u_step = []
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step
        if len(self.out) - len(self.u) < 0:
            observation = [-1000, -1000]
        else:
            observation = [np.sum(np.square(np.array(self.out)-self.u)), np.sum(np.array(self.out)-self.u)]
        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0

        return np.array(observation).astype(np.float32)

    def step(self, action):
        # get p and i from action
        action = (action + 1) * 100
        controller_p = action[0]
        controller_i = action[1]
        # controller_i = 0
        # print(action)


        # close loop with pi controller with given parameters
        pi_controller = control.tf([controller_p, controller_i], [1, 0])
        open_loop = control.series(pi_controller, self.sys)
        closed_loop = control.feedback(open_loop, 1, -1)

        # simulate one controller update step
        i = self.simulation_time_steps
        # try:
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.forced_response(closed_loop,
                                                                 self.t[i:i + self.model_steps_per_controller_value+1],
                                                                 self.u[i:i + self.model_steps_per_controller_value+1],
                                                                 return_x=True)
        else:
            sim_time, out_step, self.last_state = control.forced_response(closed_loop,
                                                                 self.t[i:i + self.model_steps_per_controller_value+1],
                                                                 self.u[i:i + self.model_steps_per_controller_value+1],
                                                                 X0=self.last_state[:, -1],
                                                                 return_x=True)
        self.out = self.out + out_step.tolist()[:-1]
        self.simulation_time_steps += self.model_steps_per_controller_value
        if self.simulation_time_steps >= len(self.t):
            self.out = self.out + self.out[-1:]
            done = True
            reward = np.mean(np.square(np.array(self.out) - self.u))
            reward = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps - 1]) - np.array(
                self.u[i:self.simulation_time_steps - 1])))
            # reward = reward / len(self.t)
            # self.render()
        else:
            done = False
            reward = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps-1]) - np.array(self.u[i:self.simulation_time_steps-1])))
            # reward = 1
        observation = self.out[self.simulation_time_steps - 1] - self.u[self.simulation_time_steps - 1]
        self.integrated_error += observation * 1/self.controller_sample_frequency
        # except ValueError:
        #     reward = -np.inf
        #     done = True
        #     observation = -np.inf
        # reward = np.clip(reward, -50, 0)


        # Optionally we can pass additional info, we are not using that for now
        info = {}
        reward = 1 - reward
        # just for logging
        self.actions_log.append([controller_p, controller_i])
        self.rewards_log.append(reward)
        self.observations_log.append([observation, self.integrated_error])
        self.dones_log.append(done)
        self.tensorboard_log = {"Obs/Error": observation,
                                "Obs/Integrated_Error": self.integrated_error,
                                "Action/P": controller_p,
                                "Action/I": controller_i,
                                "Reward": reward,
                                "Done": 1 if done else 0}


        return np.array([observation, self.integrated_error]).astype(np.float32), reward, done, info

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


class NoControllerAdaptivePT2(gym.Env):
    def __init__(self, oscillating=False, model_sample_frequency=10_000, controller_sample_frequency=100,
                 simulation_time=1.5):
        super(NoControllerAdaptivePT2, self).__init__()
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

        self.observation_space = gym.spaces.Box(low=np.array([-1000, -1000]), high=np.array([1000, 1000]), shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-0.99]), high=np.array([1]), shape=(1,),
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
        step_height = random.uniform(4.99, 5)
        u_before_step = [0] * int(0.5 * self.model_sample_frequency)
        u_step = np.linspace(0, step_height, int(0.05 * self.model_sample_frequency)).tolist()
        # u_step = []
        u_after_step = [step_height] * int(self.n_sample_points - len(u_before_step) - len(u_step))
        self.u = u_before_step + u_step + u_after_step
        if len(self.out) - len(self.u) < 0:
            observation = [-1000, -1000]
        else:
            observation = [np.sum(np.square(np.array(self.out)-self.u)), np.sum(np.array(self.out)-self.u)]
        self.last_state = None
        self.simulation_time_steps = 0
        self.out = []
        self.integrated_error = 0

        return np.array(observation).astype(np.float32)

    def step(self, action):
        # get p and i from action
        action = action * 100

        # constant_value until next update
        reference_value = [action[0]] * (self.model_steps_per_controller_value + 1)

        # simulate one controller update step
        i = self.simulation_time_steps
        # try:
        if self.last_state is None:

            sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                 self.t[i:i + self.model_steps_per_controller_value+1],
                                                                 reference_value,
                                                                 return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                     self.t[i:i + self.model_steps_per_controller_value+1],
                                                                     reference_value,
                                                                     X0=self.last_state[:, -1],
                                                                     return_x=True)
            except ValueError:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                     self.t[i:i + self.model_steps_per_controller_value+1],
                                                                     reference_value[:-1],
                                                                     X0=self.last_state[:, -1],
                                                                     return_x=True)

        self.out = self.out + out_step.tolist()[:-1]
        self.simulation_time_steps += self.model_steps_per_controller_value
        if self.simulation_time_steps >= len(self.t):
            self.out = self.out + self.out[-1:]
            done = True
            reward = np.mean(np.square(np.array(self.out) - self.u))
            reward = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps - 1]) - np.array(
                self.u[i:self.simulation_time_steps - 1])))
            # reward = reward / len(self.t)
            # self.render()
        else:
            done = False
            reward = np.mean(np.square(np.array(self.out[i:self.simulation_time_steps-1]) - np.array(self.u[i:self.simulation_time_steps-1])))
            # reward = 1
        observation = self.out[self.simulation_time_steps - 1] - self.u[self.simulation_time_steps - 1]
        self.integrated_error += observation * 1/self.controller_sample_frequency
        # except ValueError:
        #     reward = -np.inf
        #     done = True
        #     observation = -np.inf
        # reward = np.clip(reward, -50, 0)


        # Optionally we can pass additional info, we are not using that for now
        info = {}
        reward = 1 - reward
        # just for logging
        self.actions_log.append(reference_value)
        self.rewards_log.append(reward)
        self.observations_log.append([observation, self.integrated_error])
        self.dones_log.append(done)
        self.tensorboard_log = {"Obs/Error": observation,
                                "Obs/Integrated_Error": self.integrated_error,
                                "Action/Ref": reference_value,
                                "Reward": reward,
                                "Done": 1 if done else 0}


        return np.array([observation, self.integrated_error]).astype(np.float32), reward, done, info

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

