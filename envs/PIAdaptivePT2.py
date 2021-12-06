import time
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from envs.TfSim import TfSim


class PIAdaptivePT2(gym.Env):

    def __init__(self, oscillating=True, log=False):
        super(PIAdaptivePT2, self).__init__()
        if oscillating:
            self.open_loop_sys = control.tf([1], [0.001, 0.005, 1])
        else:
            self.open_loop_sys = control.tf([1], [0.001, 0.05, 1])

        self.sim = TfSim(None, 10_000, 200, 100, 1.5)

        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.w = []
        self.w_sensor = []
        self.integrated_error = 0
        self.log = log

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []

        self.observation_space = gym.spaces.Box(low=np.array([-100]*9), high=np.array([100]*9),
                                                shape=(9,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

    def reset(self, step_height=None, step_slope=None):
        if len(self.episode_log["obs"]) != 0:
            self.log_all.append(self.episode_log)

        self.sim.reset()
        self.w, self.w_sensor = self.set_w(step_height, step_slope)
        self.integrated_error = 0
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)

        if self.log:
            self.episode_log["obs"]["last_set_points"] = []
            self.episode_log["obs"]["last_system_inputs"] = []
            self.episode_log["obs"]["last_system_outputs"] = []
            self.episode_log["obs"]["errors"] = []
            self.episode_log["obs"]["set_point"] = []
            self.episode_log["obs"]["system_input"] = []
            self.episode_log["obs"]["system_output"] = []
            self.episode_log["obs"]["set_point_vel"] = []
            self.episode_log["obs"]["input_vel"] = []
            self.episode_log["obs"]["outputs_vel"] = []
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["rewards"]["pen_error_integrated"] = []
            self.episode_log["action"]["value"] = []
            self.episode_log["action"]["change"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None


        obs = self._create_obs(first=True)
        return obs

    def set_w(self, step_height=None, step_slope=None):
        if step_height is None:
            step_height = np.random.uniform(-10, 10)
        if step_slope is None:
            step_slope = np.random.uniform(0, 0.5)
        w_before_step = [0] * int(0.5 * self.sim.model_freq)
        w_step = np.linspace(0, step_height, int(step_slope * self.sim.model_freq)).tolist()
        w_after_step = [step_height] * int(self.sim.n_sample_points - len(w_before_step) - len(w_step))
        w = w_before_step + w_step + w_after_step
        w_sensor = w[::self.sim.model_steps_per_senor_update]
        return w, w_sensor

    def _create_obs(self, first=False):
        if first:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
        else:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)

        if self.log:
            self.episode_log["obs"]["last_set_points"].append(list(self.last_set_points))
            self.episode_log["obs"]["last_system_inputs"].append(list(self.last_system_inputs))
            self.episode_log["obs"]["last_system_outputs"].append(list(self.last_system_outputs))
        return obs

    def _create_reward(self):
        y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
        w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
        e = np.mean(w - y)
        self.integrated_error = self.integrated_error + e * (1/self.sim.model_steps_per_controller_update)
        self.integrated_error = np.clip(self.integrated_error, -20, 20)

        pen_error = np.abs(e)
        pen_error = pen_error * 1
        pen_action = np.abs(list(self.last_system_inputs)[-(self.sim.sensor_steps_per_controller_update+1)]
                               - list(self.last_system_inputs)[-self.sim.sensor_steps_per_controller_update]) * 0.1
        pen_integrated = np.abs(self.integrated_error) * 0
        reward = -pen_error - pen_action - pen_integrated

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)
            self.episode_log["rewards"]["pen_error_integrated"].append(-pen_integrated)

        return reward


    def step(self, action):
        controller_p = (action[0] + 1.0000001) * 50
        controller_p = np.clip(controller_p, 0, 100)
        # get p and i from action
        controller_i = 0

        if self.log:
            self.episode_log["action"]["value"].append(controller_p)
            self.episode_log["action"]["change"].append(controller_p - self.last_system_inputs[-1])


        # close loop with pi controller with given parameters
        pi_controller = control.tf([controller_p, controller_i], [1, 0])
        open_loop = control.series(pi_controller, self.open_loop_sys)
        closed_loop = control.feedback(open_loop, 1, -1)
        # set sim sys
        self.sim.sys = control.tf2ss(closed_loop)


        self.sim.sim_one_step(u=self.w[self.sim.current_simulation_step:self.sim.current_simulation_step+self.sim.model_steps_per_controller_update+1])

        for step in range(self.sim.sensor_steps_per_controller_update, 0, -1):
            self.last_system_inputs.append(controller_p)
            self.last_system_outputs.append(self.sim.sensor_out[self.sim.current_sensor_step - step])
            self.last_set_points.append(self.w[self.sim.current_simulation_step - (step-1)*self.sim.model_steps_per_senor_update-1])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = self.w
                self.episode_log["function"]["y"] = self.sim._sim_out
        else:
            done = False

        obs = self._create_obs(first=False)
        reward = self._create_reward()

        return np.array(obs), reward, done, {}

    def render(self, mode='console'):
        ...

    def create_eval_plot(self):
        fig, ax = plt.subplots(2, 2, figsize=(20, 12))
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

        fig.tight_layout()
        return fig

