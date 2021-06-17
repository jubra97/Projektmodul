import numpy as np
import gym
from gym import spaces
import control


class DmsSim(gym.Env):
    """
    Custom Environment that follows gym interface.
    """

    def __init__(self):
        super(DmsSim, self).__init__()

        # Size of the 1D-grid
        self.sys = control.tf([5.9e11, 1.2e12], [1, 255, 1e5, 1.6e7, 2.3e9, 1.7e11, 3.7e11])
        self.t = np.linspace(0, 3.5, 10000)
        self.u = [1] * 10000

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        return np.array([1]).astype(np.float32)

    def step(self, action):
        p = action[0]
        i = action[1]

        pi_controller = control.tf([p, i], [1, 0])
        open_loop = control.series(pi_controller, self.sys)
        closed_loop = control.feedback(self.sys,1 , -1)

        out = control.forced_response(closed_loop, self.t, 1)

        # Always done
        done = True

        reward = (np.square(out[1] - self.u)).mean(ax=None)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([1]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
