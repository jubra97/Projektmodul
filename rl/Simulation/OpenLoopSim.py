import copy
import control
import numpy as np


class OpenLoopSim:
    def __init__(self, system, model_freq, sensor_freq, controller_freq, action_scale, obs_scale, simulation_time=1.5):
        """
        Create a simulation environment to step through a simulation.
        :param system: System to simulate. Must be a StateSpace or Transfer Function.
        :param model_freq: Simulation sample rate
        :param sensor_freq: Sensor update frequency.
        :param controller_freq: Input update frequency. One simulation step is from one to anther input.
        :param action_scale: Scaling for action.
        :param obs_scale: Scaling for output "observation".
        :param simulation_time: How long should the simulation run.
        """
        # check for correct input
        if model_freq % controller_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of controller_update_frequency")
        if model_freq % sensor_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of sensor_sample_frequency")

        sys = copy.deepcopy(system)
        if isinstance(sys, control.StateSpace):
            self.sys = sys
        else:
            self.sys = control.tf2ss(sys)

        self.action_scale = action_scale
        self.obs_scale = obs_scale

        self.model_freq = model_freq
        self.sensor_freq = sensor_freq
        self.controller_freq = controller_freq
        self.model_steps_per_controller_update = int(self.model_freq / self.controller_freq)
        self.model_steps_per_senor_update = int(self.model_freq / self.sensor_freq)
        self.sensor_steps_per_controller_update = int(self.sensor_freq / self.controller_freq)
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_freq * simulation_time)
        self.t = np.linspace(0, simulation_time, self.n_sample_points)

        self.current_simulation_time = 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = []
        self.sensor_out = []
        self._u = []
        self.u_sensor = []  # u at sensor out
        self.t_sensor = []
        self.done = False

    def reset(self):
        """
        Reset Simulation for a new run.
        :return:
        """
        self.current_simulation_time = 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = []
        self.sensor_out = []
        self._u = []
        self.u_sensor = []  # u at sensor out
        self.t_sensor = []
        self.done = False

    def sim_one_step(self, u, add_noise=True):
        """
        Step trough simulation with given input for len(u) steps.
        :param u: Input for the simulation. Must be as long as the simulated timesteps.
        :param add_noise: Add noise to the output?
        :return:
        """
        start = self.current_simulation_step
        stop = self.current_simulation_step + self.model_steps_per_controller_update
        u_scaled = np.array(u) * self.action_scale

        # simulate next steps
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                          self.t[start:stop + 1],
                                                                          u_scaled,
                                                                          return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[start:stop + 1],
                                                                              u_scaled,
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
            except ValueError:
                sim_time, out_step, self.last_state = control.forced_response(self.sys,
                                                                              self.t[start:stop + 1],
                                                                              u_scaled[:-1],
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
        out_step = out_step / self.obs_scale

        if add_noise:
            out_step = out_step + np.random.normal(0, 0.0005, size=out_step.shape[0])

        # check if last sim step
        if stop == len(self.t):
            self._sim_out = self._sim_out + out_step.tolist()
            self._u = self._u + u
            self.done = True
        else:
            self._sim_out = self._sim_out + out_step.tolist()[:-1]
            self._u = self._u + u[:-1]

        # get newest data and append to corresponding list, get newest data from end of list and revert than again
        new_y_sensor = self._sim_out[stop:start:-self.model_steps_per_senor_update][::-1]
        self.sensor_out = self.sensor_out + new_y_sensor

        new_u_sensor = u[:1:-self.model_steps_per_senor_update][::-1]
        self.u_sensor = self.u_sensor + new_u_sensor

        new_t_sensor = self.t[self.current_simulation_step+len(u):+self.current_simulation_step+1:-self.model_steps_per_senor_update][::-1].tolist()
        self.t_sensor = self.t_sensor + new_t_sensor

        self.current_simulation_step = stop
        self.current_simulation_time = self.t[stop - 1]
        self.current_sensor_step += self.sensor_steps_per_controller_update

        return new_t_sensor, new_u_sensor, new_y_sensor
