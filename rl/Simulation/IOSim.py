import control
import numpy as np
# np.seterr(all='raise')
import copy


class IOSim:
    def __init__(self, system, model_freq, sensor_freq, controller_freq, action_scale, obs_scale, simulation_time=1.5):
        # check for correct input
        if model_freq % controller_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of controller_update_frequency")
        if model_freq % sensor_freq != 0:
            raise ValueError("model_sample_frequency must be a multiple of sensor_sample_frequency")

        if system:
            self.sys = copy.deepcopy(system)
        else:
            self.sys = None
        self.model_freq = model_freq
        self.sensor_freq = sensor_freq
        self.controller_freq = controller_freq
        self.model_steps_per_controller_update = int(self.model_freq / self.controller_freq)
        self.model_steps_per_senor_update = int(self.model_freq / self.sensor_freq)
        self.sensor_steps_per_controller_update = int(self.sensor_freq / self.controller_freq)
        self.simulation_time = simulation_time
        self.n_sample_points = int(model_freq * simulation_time)
        self.t = np.linspace(0, simulation_time, self.n_sample_points)
        self.obs_scale = obs_scale
        self.action_scale = action_scale

        self.current_simulation_time = 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = np.array([]).reshape((4, 0))
        self.sensor_out = np.array([]).reshape((4, 0))
        self._w = np.array([]).reshape((2, 0))
        self.w_sensor = np.array([]).reshape((2, 0)) # w at sensor out
        self.t_sensor = np.array([])
        self.done = False

    def reset(self):
        self.current_simulation_time = 0
        self.current_simulation_step = 0
        self.current_sensor_step = 0
        self.last_state = None
        self._sim_out = np.array([]).reshape((4, 0))
        self.sensor_out = np.array([]).reshape((4, 0))
        self._w = np.array([]).reshape((2, 0))
        self.w_sensor = np.array([]).reshape((2, 0))
        self.done = False

    def sim_one_step(self, w):
        start = self.current_simulation_step
        stop = self.current_simulation_step + self.model_steps_per_controller_update
        if self.last_state is None:
            sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                                self.t[start:stop+1],
                                                                                w,
                                                                                return_x=True)
        else:
            try:
                sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                                    self.t[start:stop+1],
                                                                                    w,
                                                                                    X0=self.last_state[:, -1],
                                                                                    return_x=True)
            except ValueError:
                print(self.t[start:stop:+1])
                print(self.current_simulation_step)
                sim_time, out_step, self.last_state = control.input_output_response(self.sys,
                                                                                    self.t[start:stop],
                                                                                    w[:, :-1],
                                                                                    X0=self.last_state[:, -1],
                                                                                    return_x=True)

        if stop == len(self.t):
            self._sim_out = np.concatenate((self._sim_out, out_step), axis=1)
            self._w = np.concatenate((self._w, w), axis=1)
            self.done = True
        else:
            self._sim_out = np.concatenate((self._sim_out, out_step[:, :-1]), axis=1)
            self._w = np.concatenate((self._w, w[:, :-1]), axis=1)

        new_sensor_out = self._sim_out[:, stop:start:-self.model_steps_per_senor_update][:, ::-1]
        self.sensor_out = np.concatenate((self.sensor_out, new_sensor_out), axis=1)

        new_w_sensor = w[:, 1:-self.model_steps_per_senor_update][:, ::-1]
        self.w_sensor = np.concatenate((self.w_sensor, new_w_sensor), axis=1)

        new_t_sensor = self.t[stop:start:-self.model_steps_per_senor_update]
        self.t_sensor = np.concatenate((self.t_sensor, new_t_sensor))

        self.current_simulation_step = stop
        self.current_simulation_time = self.t[stop-1]
        self.current_sensor_step += self.sensor_steps_per_controller_update

        return new_t_sensor, new_w_sensor, new_sensor_out
