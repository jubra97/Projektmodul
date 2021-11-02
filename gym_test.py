import random

import numpy as np
import gym
from gym import spaces
import control
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class DmsSim(gym.Env):
    """
    Custom Environment that follows gym interface.
    """

    def __init__(self):
        super(DmsSim, self).__init__()

        # Size of the 1D-grid
        self.sys = control.tf([5.9e11, 1.2e12], [1, 255, 1e5, 1.6e7, 2.3e9, 1.7e11, 3.7e11])
        self.t = np.linspace(0, 2.5, 10000)  # simulate for 2.5 seconds with 10_000 steps
        self.u = [0] + [1] * 9999  # step to one
        self.out = None

        # init rendering
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.first_flag = True

        # possible observations (step height)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32)
        # P, I - Value; TODO: If P is 1 errors can occur; System not stable!?
        self.action_space = spaces.Box(low=-np.array([0, 0]), high=np.array([0.1, 10]), shape=(2,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array; Return after every step with random step between 0, 2
        :return: (np.array)
        """
        step_height = random.uniform(0, 2)
        self.u = [0] + [step_height] * 9999
        return np.array([step_height]).astype(np.float32)

    def step(self, action):
        p = action[0]
        i = action[1]

        # close loop with pi controller with given parameters
        pi_controller = control.tf([p, i], [1, 0])
        open_loop = control.series(pi_controller, self.sys)
        closed_loop = control.feedback(open_loop, 1, -1)

        # simulate step
        self.out = control.forced_response(closed_loop, self.t, self.u)

        # Always done
        done = True

        # error between step and simulation result; TODO: Sum or mean?; Normalize with step height?
        reward = -(np.square(self.out[1] - self.u)).mean() / self.u[-1]

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([1]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if self.first_flag:
            self.ax.set_ylim(-0.1, 2.2)
            self.line1, = self.ax.plot(self.out[0], self.out[1],  "r-")
            self.line2, = self.ax.plot(self.out[0], self.u)
            self.ax.grid()
            self.first_flag = False
        else:
            self.line1.set_ydata(self.out[1])
            self.line2.set_ydata(self.u)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        pass


if __name__ == "__main__":
    test_sim = DmsSim()
    test_sim.step([0.99, 1])