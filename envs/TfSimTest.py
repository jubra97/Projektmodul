import control
import numpy as np
np.seterr(all='raise')
import copy


class TfSim:
    def __init__(self, system, model_freq, sensor_freq, controller_freq, simulation_time=1.5):
        # check for correct input
        if model_freq % controller_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of controller_update_frequency")
        if model_freq % sensor_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of sensor_sample_frequency")

        # if system:
        #     self.sys = copy.deepcopy(system)
        #     self.sys = control.tf2ss(self.sys)
        # else:
        #     self.sys = None
        self.sys = system
        self.model_freq = model_freq
        self.sensor_freq = sensor_freq
        self.controller_freq = controller_freq
        self.model_steps_per_controller_update = int(self.model_freq / self.controller_freq)
        self.model_steps_per_senor_update = int(self.model_freq / self.sensor_freq)
        self.sensor_steps_per_controller_update = int(self.sensor_freq / self.controller_freq)
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_freq * simulation_time)
        self.t = np.linspace(0.00000000000000000000000001, simulation_time, self.n_sample_points)

        self.current_simulation_time = 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = np.array([]).reshape((4, 0))
        self.sensor_out = []
        self._u = np.array([]).reshape((2, 0))
        self.u_sensor = []  # u at sensor out
        self.done = False

    def reset(self):
        self.current_simulation_time= 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = []
        self.sensor_out = []
        self._u = []
        self.u_sensor = []  # u at sensor out
        self.done = False

    def sim_one_step(self, u, add_noise=True):
        start = self.current_simulation_step
        stop = self.current_simulation_step + self.model_steps_per_controller_update
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                          self.t[start:stop+1],
                                                                          u,
                                                                          return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                              self.t[start:stop+1],
                                                                              u,
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
            except IndexError:
                sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                              self.t[start:stop+1],
                                                                              u[:-1],
                                                                              X0=self.last_state[:, -1],
                                                                              return_x=True)
        if add_noise:
            out_step = out_step + np.random.normal(0, 0.005, size=out_step.shape[0])

        if stop == len(self.t):
            self._sim_out = np.concatenate((self._sim_out, out_step))
            self._u = np.concatenate((self._u, u))
            self.done = True
        else:
            self._sim_out = np.concatenate((self._sim_out, out_step[:, :-1]), axis=1)
            self._u = np.concatenate((self._u, u[:, :-1]), axis=1)

        self.sensor_out = self.sensor_out + self._sim_out[stop:start:-self.model_steps_per_senor_update][::-1]
        self.u_sensor = self.u_sensor + u[:1:-self.model_steps_per_senor_update][::-1]


        self.current_simulation_step = stop
        self.current_simulation_time = self.t[stop-1]
        self.current_sensor_step += self.sensor_steps_per_controller_update



    def sim_all(self, u, add_noise=True):
        sim_time, out_step = control.forced_response(self.sys, self.t, u)

        if add_noise:
            out_step = out_step + np.random.normal(0, 0.01, size=out_step.shape[0])

        self._sim_out = out_step.tolist()
        self.done = True

        self.sensor_out = self._sim_out[::self.model_steps_per_senor_update]
        self.u_sensor = self._u[::self.model_steps_per_senor_update]

        self.current_simulation_step = len(self.t) - 1
        self.current_simulation_time = self.t[-1]
        self.current_sensor_step = len(self.u_sensor)