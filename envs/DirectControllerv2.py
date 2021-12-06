import time
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from envs.TfSim import TfSim


class DirectControllerPT2(gym.Env):

    def __init__(self, oscillating=False, log=False):
        super(DirectControllerPT2, self).__init__()
        if oscillating:
            sys = control.tf([1], [0.001, 0.005, 1])
        else:
            sys = control.tf([1], [0.001, 0.05, 1])

        self.sim = TfSim(sys, 10_000, 200, 100, 1.5)

        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.w = []
        self.w_sensor = []
        self.integrated_error = 0
        self.log = log
        self.n_episodes = 0

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []

        self.observation_space = gym.spaces.Box(low=np.array([-100]*6), high=np.array([100]*6),
                                                shape=(6,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,),
                                           dtype=np.float32)

    def reset(self, step_height=None, step_slope=None):
        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1

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


        obs = self._create_obs_with_vel(first=True)
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

    def _create_obs_with_vel(self, first=False):
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        system_inputs = np.array(list(self.last_system_inputs))

        outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1/self.sim.sensor_steps_per_controller_update
        input_vel = (system_inputs[-3] - system_inputs[-1]) * 1/self.sim.model_steps_per_controller_update
        set_point_vel = (set_points[-2] - set_points[-1]) * 1/self.sim.sensor_steps_per_controller_update

        if first:
            obs = [set_points[-1], system_inputs[-1], system_outputs[-1], set_point_vel, input_vel, outputs_vel]
        else:
            obs = [set_points[-1], system_inputs[-1], system_outputs[-1], set_point_vel, input_vel, outputs_vel]

        if self.log:
            self.episode_log["obs"]["set_point"].append(set_points[-1])
            self.episode_log["obs"]["system_input"].append(system_inputs[-1])
            self.episode_log["obs"]["system_output"].append(system_outputs[-1])
            self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
            self.episode_log["obs"]["input_vel"].append(input_vel)
            self.episode_log["obs"]["outputs_vel"].append(outputs_vel)
        return obs

    def _create_obs_last_states(self, first=False):
        if first:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
        else:
            obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)

        if self.log:
            self.episode_log["obs"]["last_set_points"].append(list(self.last_set_points))
            self.episode_log["obs"]["last_system_inputs"].append(list(self.last_system_inputs))
            self.episode_log["obs"]["last_system_outputs"].append(list(self.last_system_outputs))
        return obs

    def _create_obs_last_errors(self, first=False):
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        errors = (set_points - system_outputs).tolist()

        if first:
            obs = errors + list(self.last_system_inputs)
        else:
            obs = errors + list(self.last_system_inputs)

        if self.log:
            self.episode_log["obs"]["errors"].append(errors)
            self.episode_log["obs"]["last_system_inputs"].append(list(self.last_system_inputs))
        return obs

    def _create_reward(self):
        y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
        w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
        e = np.mean(w - y)
        self.integrated_error = self.integrated_error + e * (1/self.sim.model_steps_per_controller_update)
        self.integrated_error = np.clip(self.integrated_error, -20, 20)

        pen_error = np.abs(e)
        pen_error = pen_error * 1
        pen_action = np.square(list(self.last_system_inputs)[-(self.sim.sensor_steps_per_controller_update+1)]
                               - list(self.last_system_inputs)[-self.sim.sensor_steps_per_controller_update]) * 0.5
        pen_integrated = np.abs(self.integrated_error) * 0.2
        reward = -pen_error - pen_action - pen_integrated

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["pen_action"].append(-pen_action)
            self.episode_log["rewards"]["pen_error_integrated"].append(-pen_integrated)

        return reward

    def step(self, action):
        # action und obs oder last action und obs? Schaue ich sonst in die Zukunft?
        system_input = np.clip(action[0] * 20, -20, 20)
        # system_input = round(system_input, 1)
        system_input_trajectory = [system_input] * (self.sim.model_steps_per_controller_update + 1)

        if self.log:
            self.episode_log["action"]["value"].append(system_input)
            self.episode_log["action"]["change"].append(system_input - self.last_system_inputs[-1])

        self.sim.sim_one_step(u=system_input_trajectory, add_noise=True)

        # if self.sim.current_simulation_step != 0:
        for step in range(self.sim.sensor_steps_per_controller_update, 0, -1):
            self.last_system_inputs.append(self.sim.u_sensor[self.sim.current_sensor_step - step])
            self.last_system_outputs.append(self.sim.sensor_out[self.sim.current_sensor_step - step])
            self.last_set_points.append(self.w[self.sim.current_simulation_step - (step-1)*self.sim.model_steps_per_senor_update-1])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = self.w
                self.episode_log["function"]["y"] = self.sim._sim_out
        else:
            done = False

        obs = self._create_obs_with_vel(first=False)
        reward = self._create_reward()

        return np.array(obs), reward, done, {}

    def render(self, mode="human"):
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
