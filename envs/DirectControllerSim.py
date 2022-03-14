from collections import deque

import control
import numpy as np

from Simulation.OpenLoopSim import OpenLoopSim
from envs.DirectControl import DirectController


class DirectControllerSim(DirectController):
    def __init__(self, log=False):
        sys = control.tf([3.55e3], [0.00003, 0.0014, 1])  # pt2 of dms
        # create simulation object with an arbitrary tf.
        self.sim = OpenLoopSim(sys, 12_000, 4000, 100, action_scale=500, obs_scale=3_000_000, simulation_time=1.5)
        self.sys_gain = (self.sim.sys.C @ np.linalg.inv(-self.sim.sys.A) @ self.sim.sys.B + self.sim.sys.D)[0][0]

        super().__init__(log=log, sensor_freq=4000)

    def custom_reset(self, step_start=None, step_end=None, step_slope=None, custom_w=None):
        # reset simulation
        self.sim.reset()

        # create reference value (w)
        if custom_w is not None:
            assert self.sim.n_sample_points == len(custom_w), "Simulation and input length must not differ"
            self.w = custom_w
        else:
            self.w = self.set_w(step_start, step_end, step_slope)

        # set x0 in state space, to do so compute step response to given w[0]
        T = control.timeresp._default_time_vector(self.sim.sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        U = np.ones_like(T) * self.w[0] * (self.sim.obs_scale / self.sys_gain)
        _, step_response, states = control.forced_response(self.sim.sys, T, U, return_x=True)
        self.sim.last_state = np.array([states[:, -1]]).T

        initial_input = ((self.w[0] * self.sim.obs_scale) / (self.sys_gain * self.sim.action_scale))
        self.last_u = deque([initial_input] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        initial_output = (self.sim.last_state[:, -1] @ self.sim.sys.C.T)[0] / self.sim.obs_scale
        self.last_y = deque([initial_output] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([self.w[0]] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

    def set_w(self, step_start=None, step_end=None, step_slope=None):
        """
        Create reference value (w) as ramp or step
        :param step_height: Height of ramp/ step
        :param step_slope: Slope of ramp; if 0 a step is generated
        :return:
        """
        if step_start is None:
            step_start = np.random.uniform(0, 0.5)
        if step_end is None:
            step_end = np.random.uniform(0, 0.5)
        if step_slope is None:
            step_slope = np.random.uniform(0, 0.5)
        w_before_step = [step_start] * int(0.5 * self.sim.model_freq)
        w_step = np.linspace(w_before_step[0], step_end, int(step_slope * self.sim.model_freq)).tolist()
        w_after_step = [step_end] * int(self.sim.n_sample_points - len(w_before_step) - len(w_step))
        w = w_before_step + w_step + w_after_step
        return w

    def update_simulation(self, u_trajectory):
        w_current_sim_step = self.w[self.sim.current_simulation_step+len(u_trajectory):self.sim.current_simulation_step+1:-self.sim.model_steps_per_senor_update][::-1]
        # simulate system until next update of u.
        t, u, y = self.sim.sim_one_step(u=u_trajectory, add_noise=True)

        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time is used for more precise simulation results.
        # Sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # For the fifo lists the newest sensor values are used.

        for step in range(len(t)):
            self.last_t.append(t[step])
            self.last_u.append(u[step])
            self.last_y.append(y[step])
            self.last_w.append(w_current_sim_step[step])


    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return:
        """
        # action[0] = action[0] ** 2
        # create static input for every simulation step until next update of u.

        new_action = self.last_u[-1] + action[0]
        new_action = np.clip(new_action, -1, 1)
        system_input_trajectory = [new_action] * (self.sim.model_steps_per_controller_update + 1)
        self.update_simulation(system_input_trajectory)

        if self.log:
            self.episode_log["action"]["value"].append(action[0])
            self.episode_log["action"]["change"].append(action[0] - self.last_u[-1])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = np.array(self.w) * self.sim.obs_scale
                self.episode_log["function"]["y"] = np.array(self.sim._sim_out) * self.sim.obs_scale
        else:
            done = False

        # create obs and reward
        obs = self.observation_function()
        reward = self.reward_function()
        return obs, reward, done, {}
