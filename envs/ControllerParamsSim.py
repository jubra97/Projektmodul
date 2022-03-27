from collections import deque

import control
import numpy as np

from Simulation.IOSim import IOSim
from envs.ControllerParams import ControllerParams


class ControllerParamsSim(ControllerParams):
    def __init__(self,
                 log=False,
                 model_freq=8_000,
                 output_freq=100,
                 sensor_freq=4000,
                 obs_config=None,
                 reward_function="discrete_u_pen_dep_on_error",
                 observation_function="error_with_vel",
                 oscillation_pen_gain=0.01,
                 oscillation_pen_fun=np.sqrt,
                 error_pen_fun=None
                 ):
        self.open_loop_sys = control.tf2ss(control.tf([1], [0.00003, 0.0014, 1]))  # pt2 of dms
        # create simulation object with an arbitrary tf.
        self.sim = IOSim(None, model_freq, sensor_freq, output_freq, action_scale=1, obs_scale=1, simulation_time=1.5)
        # self.open_loop_gain = (self.open_loop_sys.C @ np.linalg.inv(-self.open_loop_sys.A) @ self.open_loop_sys.B + self.open_loop_sys.D)[0][0]
        self.open_loop_gain = 1
        super().__init__(log=log,
                         sensor_freq=sensor_freq,
                         output_freq=output_freq,
                         obs_config=obs_config,
                         reward_function=reward_function,
                         observation_function=observation_function,
                         oscillation_pen_gain=oscillation_pen_gain,
                         oscillation_pen_fun=oscillation_pen_fun,
                         error_pen_fun=error_pen_fun,
                         )

        self.first_step = True
        self.last_p = 1

    def custom_reset(self, step_start=None, step_end=None, step_slope=None, custom_w=None):
        # reset simulation
        self.sim.reset()

        # create reference value (w)
        if custom_w is not None:
            assert self.sim.n_sample_points == len(custom_w), "Simulation and input length must not differ"
            self.w = custom_w
        else:
            self.w = self.set_w(0, step_end, step_slope)
        self.first_step = True

        # set x0 in state space, to do so compute step response to given w[0]
        T = control.timeresp._default_time_vector(self.open_loop_sys)
        # compute system gain
        # https://math.stackexchange.com/questions/2424383/how-should-i-interpret-the-static-gain-from-matlabs-command-zpkdata
        U = np.ones_like(T) * self.w[0, 0] * (self.sim.obs_scale / self.open_loop_gain)
        _, step_response, states = control.forced_response(self.open_loop_sys, T, U, return_x=True)
        self.sim.last_state = np.concatenate((np.array([[0]]), states[:, -1:]))

        initial_input = (self.w[0, 0] * self.sim.obs_scale)/(self.open_loop_gain * self.sim.action_scale)
        self.last_u = deque([initial_input] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        initial_output = (states[:, -1] @ self.open_loop_sys.C.T)[0] / self.sim.obs_scale
        self.last_y = deque([initial_output] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_w = deque([self.w[0, 0]] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)
        self.last_t = deque([0] * self.nbr_measurements_to_keep, maxlen=self.nbr_measurements_to_keep)

        self.last_p = 1

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

        noise = np.random.normal(0, 0.001, self.sim.n_sample_points)
        sys_input = np.array([w, noise])
        return sys_input

    def update_simulation(self):
        print(self.sim.last_state[:, -1])
        w_current_sim_step = self.w[:, self.sim.current_simulation_step:self.sim.current_simulation_step+self.sim.model_steps_per_controller_update+1]
        # simulate system until next update of u.
        t, w, sim_out_sensor = self.sim.sim_one_step(w=w_current_sim_step)
        u = sim_out_sensor[2, :]
        y = sim_out_sensor[1, :]
        # update fifo lists with newest values. In simulation a simulation sample time, a sensor sample time, and a u
        # update sample time is used. A smaller simulation sample time is used for more precise simulation results.
        # Sensor sample time is used to simulate that a potential sensor is slower than the system dynamics.
        # The u update sample time is used that a potential actor / calculation of u is even slower than the sensor.
        # For the fifo lists the newest sensor values are used.

        for step in range(len(t)):
            self.last_t.append(t[step])
            self.last_u.append(u[step])
            self.last_y.append(y[step])
            self.last_w.append(w[0, step])

    def step(self, action):
        """
        Step the environment for one update step of the action. Collect sensor values of the simulation and compute
        reward and next observation from them.
        :param action: Next u for simulation.
        :return:
        """
        # action[0] = action[0] ** 2
        # create static input for every simulation step until next update of u.

        new_p = (action[0] + 1) * 50
        new_p = np.clip(new_p, 0, 100)
        # system_input_trajectory = [new_action] * (self.sim.model_steps_per_controller_update + 1)
        self.sim.sys = self.create_closed_loop_io_sys(new_p)
        self.update_simulation()

        if self.log:
            self.episode_log["action"]["value"].append(new_p)
            self.episode_log["action"]["change"].append(action[0])

        if self.sim.done:
            done = True
            if self.log:
                self.episode_log["function"]["w"] = self.w[0, :]
                self.episode_log["function"]["y"] = self.sim._sim_out[1, :]
                self.episode_log["function"]["u"] = self.sim._sim_out[2, :]
                self.episode_log["function"]["e"] = self.sim._sim_out[3, :]
        else:
            done = False

        # create obs and reward
        obs = self.observation_function()
        reward = self.reward_function()
        return obs, reward, done, {}

    def create_closed_loop_io_sys(self, p, i=2, d=None):
        print(p)
        if p == 0:
            p = 1e-20
        controller_p = control.tf([p], [1])
        if i is not None:
            controller_i = control.tf([i], [1, 0])
        else:
            controller_i = 0
        if d is not None:
            controller_d = control.tf([d, 0], [1])
        else:
            controller_d = 0

        controller = controller_p + controller_i + controller_d


        # p = p * self.sim.action_scale
        # i = i * self.sim.action_scale
        self.last_p = p
        self.last_i = i
        self.last_d = d
        print(controller)
        controller = control.tf2ss(controller)
        if self.first_step:
            self.sim.last_state[0, 0] = self.w[0, 0] / (i * self.open_loop_gain)
            self.first_step = False

        io_open_loop = control.LinearIOSystem(self.open_loop_sys, inputs="u", outputs="y", name="open_loop")
        io_controller = control.LinearIOSystem(controller, inputs="e", outputs="u", name="controller")
        w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
        y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

        closed_loop = control.interconnect([w_y_comp, io_controller, io_open_loop, y_noise], name="closed_loop",
                                           inplist=["w", "noise"],
                                           outlist=["y", "y_noisy", "u", "e"])
        print(closed_loop.A)
        print(closed_loop.B)
        print(closed_loop.C)
        print(closed_loop.D)
        return closed_loop