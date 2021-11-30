import unittest
from envs.TfSim import TfSim
import control
import numpy as np
import matplotlib.pyplot as plt


class TestSimulation(unittest.TestCase):

    def test_step_simulation_open_loop(self):
        sys = control.tf([1], [0.001, 0.005, 1])
        tf_sim = TfSim(sys, 10_000, 200, 100)

        u = [0] * 5000 + [1] * 10_000
        t = tf_sim.t
        sys = tf_sim.sys

        _, out_step_comp = control.forced_response(sys, t, u)

        while not tf_sim.done:
            tf_sim.sim_one_step(u=u[tf_sim.current_simulation_step:tf_sim.current_simulation_step+tf_sim.model_steps_per_controller_update+1], add_noise=False)

        out_step_comp = np.array(out_step_comp)
        out_step = np.array(tf_sim._sim_out)

        sim_is_same = np.allclose(out_step, out_step_comp)
        self.assertTrue(sim_is_same)

    def test_sensor_sensor_out(self):
        sys = control.tf([1], [0.001, 0.005, 1])
        tf_sim1 = TfSim(sys, 10_000, 200, 100)
        tf_sim2 = TfSim(sys, 10_000, 200, 100)

        u = [0] * 5000 + [1] * 10_000

        tf_sim1.sim_all(u=u, add_noise=False)
        tf_sim1_sensor_sampled = tf_sim1._sim_out[::tf_sim1.model_steps_per_senor_update]
        tf_sim1_sensor_u_sampled = tf_sim1._u[::tf_sim1.model_steps_per_senor_update]

        while not tf_sim2.done:
            tf_sim2.sim_one_step(u=u[tf_sim2.current_simulation_step:tf_sim2.current_simulation_step+tf_sim2.model_steps_per_controller_update+1], add_noise=False)
        tf_sim2_sensor_sampled = tf_sim2.sensor_out
        tf_sim2_sensor_u_sampled = tf_sim2.u_sensor

        tf_sim1_sensor_sampled = np.array(tf_sim1_sensor_sampled)
        tf_sim2_sensor_sampled = np.array(tf_sim2_sensor_sampled)
        same_out = np.allclose(tf_sim1_sensor_sampled, tf_sim2_sensor_sampled)
        self.assertTrue(same_out)

        tf_sim1_sensor_u_sampled = np.array(tf_sim1_sensor_u_sampled)
        tf_sim2_sensor_u_sampled = np.array(tf_sim2_sensor_u_sampled)

        same_out = np.allclose(tf_sim1_sensor_u_sampled, tf_sim2_sensor_u_sampled)
        self.assertTrue(same_out)
