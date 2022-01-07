import time
from collections import deque

import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from envs.IOSim import IOSim


class PIAdaptivePT2(gym.Env):

    def __init__(self, oscillating=False, log=False):
        super(PIAdaptivePT2, self).__init__()
        if oscillating:
            self.open_loop_sys = control.tf([1], [0.001, 0.005, 1])
        else:
            self.open_loop_sys = control.tf([1], [0.001, 0.05, 1])

        self.open_loop_sys = control.tf2ss(self.open_loop_sys)
        self.sim = IOSim(None, 4_000, 200, 100, 1.5)

        self.last_ps = deque([0] * 3, maxlen=3)
        self.last_is = deque([0] * 3, maxlen=3)
        self.last_system_inputs = deque([0] * 3, maxlen=3)
        self.last_system_outputs = deque([0] * 3, maxlen=3)
        self.last_set_points = deque([0] * 3, maxlen=3)
        self.w = []
        self.integrated_error = 0
        self.abs_integrated_error = 0
        self.log = log
        self.n_episodes = 0

        self.episode_log = {"obs": {}, "rewards": {}, "action": {}, "function": {}}
        self.log_all = []

        self.observation_space = gym.spaces.Box(low=np.array([-100]*6), high=np.array([100]*6),
                                                shape=(6,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1]*1), high=np.array([1]*1), shape=(1,),
                                           dtype=np.float32)

    def reset(self, step_height=None, step_slope=None):
        if self.n_episodes != 0:
            self.log_all.append(self.episode_log)
        self.n_episodes += 1

        self.sim.reset()
        self.w = self.set_w(step_height, step_slope)
        self.integrated_error = 0
        self.abs_integrated_error = 0
        self.last_ps = deque([0] * 3, maxlen=3)
        self.last_is = deque([0] * 3, maxlen=3)
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
            self.episode_log["obs"]["integrated_error"] = []
            self.episode_log["obs"]["p"] = []
            self.episode_log["obs"]["i"] = []
            self.episode_log["rewards"]["summed"] = []
            self.episode_log["rewards"]["pen_error"] = []
            self.episode_log["rewards"]["pen_action"] = []
            self.episode_log["rewards"]["u_change"] = []
            self.episode_log["rewards"]["pen_error_integrated"] = []
            self.episode_log["action"]["p_value"] = []
            self.episode_log["action"]["p_change"] = []
            self.episode_log["action"]["i_value"] = []
            self.episode_log["action"]["i_change"] = []
            self.episode_log["function"]["w"] = None
            self.episode_log["function"]["y"] = None


        obs = self.create_obs(first=True)
        return obs

    def set_w(self, step_height=None, step_slope=None, add_noise=True):
        if step_height is None:
            step_height = np.random.uniform(-10, 10)
        if step_slope is None:
            step_slope = np.random.uniform(0, 0.5)
        w_before_step = [0] * int(0.5 * self.sim.model_freq)
        w_step = np.linspace(0, step_height, int(step_slope * self.sim.model_freq)).tolist()
        w_after_step = [step_height] * int(self.sim.n_sample_points - len(w_before_step) - len(w_step))
        w = w_before_step + w_step + w_after_step

        if add_noise:
            noise = np.random.normal(0, 0.001, 6_000)
        else:
            noise = [0] * 6_000

        sys_input = np.array([w, noise])

        return sys_input

    # def _create_obs_with_vel(self, first=False):
    #     set_points = np.array(list(self.last_set_points))
    #     system_outputs = np.array(list(self.last_system_outputs))
    #     system_inputs = np.array(list(self.last_system_inputs))
    #
    #     outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1/self.sim.sensor_steps_per_controller_update
    #     set_point_vel = (set_points[-2] - set_points[-1]) * 1/self.sim.sensor_steps_per_controller_update
    #
    #     if first:
    #         obs = [set_points[-1], *system_inputs[-1], system_outputs[-1], set_point_vel, outputs_vel, self.integrated_error]
    #     else:
    #         obs = [set_points[-1], *system_inputs[-1], system_outputs[-1], set_point_vel, outputs_vel, self.integrated_error]
    #
    #     if self.log:
    #         self.episode_log["obs"]["set_point"].append(set_points[-1])
    #         self.episode_log["obs"]["system_input"].append(system_inputs[-1])
    #         self.episode_log["obs"]["system_output"].append(system_outputs[-1])
    #         self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
    #         self.episode_log["obs"]["outputs_vel"].append(outputs_vel)
    #         self.episode_log["obs"]["integrated_error"].append(self.integrated_error)
    #     return obs
    #
    # def _create_obs_last_states(self, first=False):
    #     if first:
    #         obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
    #     else:
    #         obs = list(self.last_set_points) + list(self.last_system_inputs) + list(self.last_system_outputs)
    #
    #     if self.log:
    #         self.episode_log["obs"]["last_set_points"].append(list(self.last_set_points))
    #         self.episode_log["obs"]["last_system_inputs"].append(list(self.last_system_inputs))
    #         self.episode_log["obs"]["last_system_outputs"].append(list(self.last_system_outputs))
    #     return obs
    #
    # def _create_obs_last_errors(self, first=False):
    #     set_points = np.array(list(self.last_set_points))
    #     system_outputs = np.array(list(self.last_system_outputs))
    #     errors = (set_points - system_outputs).tolist()
    #
    #     if first:
    #         obs = errors + list(self.last_system_inputs) + [self.integrated_error]
    #     else:
    #         obs = errors + list(self.last_system_inputs) + [self.integrated_error]
    #
    #     if self.log:
    #         self.episode_log["obs"]["errors"].append(errors)
    #         self.episode_log["obs"]["last_system_inputs"].append(list(self.last_system_inputs))
    #         self.episode_log["obs"]["integrated_error"].append(self.integrated_error)
    #     return obs
    #
    # def _create_reward(self):
    #     y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
    #     w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
    #     e = np.mean(w - y)
    #     self.integrated_error = self.integrated_error + e * (1/self.sim.model_steps_per_controller_update)
    #     self.integrated_error = np.clip(self.integrated_error, -0.3, 0.3)
    #     self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.sim.model_steps_per_controller_update)
    #     self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)
    #
    #
    #     pen_error = np.abs(e)
    #     pen_error = np.sqrt(pen_error) * 1
    #     pen_integrated = np.square(self.integrated_error) * 50
    #
    #     reward = pen_error + pen_integrated
    #     reward = -reward*5
    #
    #     if self.log:
    #         self.episode_log["rewards"]["summed"].append(reward)
    #         self.episode_log["rewards"]["pen_error"].append(-pen_error)
    #         self.episode_log["rewards"]["pen_error_integrated"].append(-pen_integrated)
    #
    #     return reward
    #
    # def _create_reward_discrete(self):
    #     y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
    #     w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
    #     e = np.mean(w - y)
    #     self.integrated_error = self.integrated_error + e * (1/self.sim.model_steps_per_controller_update)
    #     self.integrated_error = np.clip(self.integrated_error, -20, 20)
    #     self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.sim.model_steps_per_controller_update)
    #     self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)
    #
    #     pen_action = np.abs(list(self.last_system_inputs)[-(self.sim.sensor_steps_per_controller_update+1)]
    #                            - list(self.last_system_inputs)[-self.sim.sensor_steps_per_controller_update])
    #
    #     pen_action = np.clip(pen_action, 0, 2) * 10
    #
    #     abs_error = np.abs(e)
    #     reward = 0
    #     if abs_error < 5:
    #         reward += 1
    #     if abs_error < 1:
    #         reward += 2
    #     if abs_error < 0.1:
    #         reward += 5
    #     if abs_error < 0.05:
    #         reward += 10
    #     if abs_error < 0.02:
    #         reward += 15
    #     if abs_error < 0.005:
    #         reward += 20
    #     reward -= pen_action + (abs_error * 0.2)
    #
    #     if self.log:
    #         self.episode_log["rewards"]["summed"].append(reward)
    #         self.episode_log["rewards"]["pen_action"].append(-pen_action)
    #         self.episode_log["rewards"]["pen_error"].append(-abs_error * 0.2)
    #     return reward

    def create_obs(self, first):
        set_points = np.array(list(self.last_set_points))
        system_outputs = np.array(list(self.last_system_outputs))
        system_inputs = np.array(list(self.last_system_inputs))
        p_s = np.array(list(self.last_ps))
        i_s = np.array(list(self.last_is))

        outputs_vel = (system_outputs[-2] - system_outputs[-1]) * 1/self.sim.sensor_steps_per_controller_update
        set_point_vel = (set_points[-2] - set_points[-1]) * 1/self.sim.sensor_steps_per_controller_update

        if first:
            obs = [p_s[-1], i_s[-1], set_points[-1], system_outputs[-1], set_point_vel, outputs_vel]
        else:
            obs = [p_s[-1], i_s[-1], set_points[-1], system_outputs[-1], set_point_vel, outputs_vel]

        if self.log:
            self.episode_log["obs"]["set_point"].append(set_points[-1])
            # self.episode_log["obs"]["system_input"].append(system_inputs[-1])
            self.episode_log["obs"]["system_output"].append(system_outputs[-1])
            self.episode_log["obs"]["set_point_vel"].append(set_point_vel)
            self.episode_log["obs"]["outputs_vel"].append(outputs_vel)
            self.episode_log["obs"]["p"].append(p_s[-1])
            self.episode_log["obs"]["i"].append(i_s[-1])
        return obs


    def _create_reward(self):
        y = np.array(list(self.last_system_outputs)[-self.sim.sensor_steps_per_controller_update:])
        w = np.array(list(self.last_set_points)[-self.sim.sensor_steps_per_controller_update:])
        e = np.mean(w - y)
        u = np.array(list(self.last_system_inputs)[-2:])
        u_change = abs(u[-1] - u[-2])

        self.integrated_error = self.integrated_error + e * (1/self.sim.model_steps_per_controller_update)
        self.integrated_error = np.clip(self.integrated_error, -0.3, 0.3)
        self.abs_integrated_error = self.abs_integrated_error + abs(e) * (1/self.sim.model_steps_per_controller_update)
        self.abs_integrated_error = np.clip(self.abs_integrated_error, 0, 20)


        pen_error = np.abs(e)
        pen_error = (pen_error) * 1
        pen_u_change = u_change * 0.05
        pen_integrated = np.square(self.integrated_error) * 0

        reward = - pen_error - pen_u_change

        if self.log:
            self.episode_log["rewards"]["summed"].append(reward)
            self.episode_log["rewards"]["pen_error"].append(-pen_error)
            self.episode_log["rewards"]["u_change"].append(-pen_u_change)
            # self.episode_log["rewards"]["pen_error_integrated"].append(u[-1])

        return reward


    def step(self, action):
        controller_p = (action[0] + 1.0000001) * 50
        # controller_i = (action[1] + 1.0000001) * 50
        controller_i = 0
        controller_p = np.clip(controller_p, 0, 100)
        controller_i = np.clip(controller_i, 0, 0)


        if self.log:
            self.episode_log["action"]["p_value"].append(controller_p)
            self.episode_log["action"]["p_change"].append(controller_p - self.last_ps[-1])
            self.episode_log["action"]["i_value"].append(controller_i)
            self.episode_log["action"]["i_change"].append(controller_i - self.last_is[-1])

        # update system with new p and i; simulate next time step
        self.sim.sys = self.create_io_sys(controller_p, controller_i)
        self.sim.sim_one_step(w=self.w[:, self.sim.current_simulation_step:self.sim.current_simulation_step+self.sim.model_steps_per_controller_update+1])

        # get values at sensor time steps
        for step in range(self.sim.sensor_steps_per_controller_update, 0, -1):
            self.last_ps.append(controller_p)
            self.last_is.append(controller_i)
            self.last_system_inputs.append(self.sim.sensor_out[2, self.sim.current_sensor_step - step])  # u
            self.last_system_outputs.append(self.sim.sensor_out[1, self.sim.current_sensor_step - step])  # y
            self.last_set_points.append(self.w[0, self.sim.current_simulation_step - (step-1)*self.sim.model_steps_per_senor_update-1])  # w

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = self.w[0, :]
                self.episode_log["function"]["y"] = self.sim._sim_out[1, :]
                self.episode_log["function"]["u"] = self.sim._sim_out[2, :]
                self.episode_log["function"]["e"] = self.sim._sim_out[3, :]
        else:
            done = False

        obs = self.create_obs(first=False)
        reward = self._create_reward()

        return np.array(obs), reward, done, {}

    def create_io_sys(self, p, i):
        pi_controller = control.tf2ss(control.tf([p, i], [1, 0]))

        io_open_loop = control.LinearIOSystem(self.open_loop_sys, inputs="u", outputs="y", name="open_loop")
        io_pi = control.LinearIOSystem(pi_controller, inputs="e", outputs="u", name="controller")
        w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
        y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

        closed_loop = control.interconnect([w_y_comp, io_pi, io_open_loop, y_noise], name="closed_loop",
                                           inplist=["w", "noise"],
                                           outlist=["y", "y_noisy", "u", "e"])
        return closed_loop

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
