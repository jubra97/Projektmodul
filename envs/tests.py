import unittest
from envs.TFGymEnv import TFGymEnv
import control
import numpy as np

class TestSimulation(unittest.TestCase):

    def test_step_simulation_out(self):
        env = TFGymEnv()
        env.reset(1, 0)
        t = env.t
        u = env.u
        sys = env.sys

        _, out_step_comp = control.forced_response(sys, t, u)

        while env.current_simulation_step < len(env.t):
            start = env.current_simulation_step
            stop = env.current_simulation_step + env.model_steps_per_controller_update
            env.sim_one_step(env.u[start:stop+1], start, stop, add_noise=False)
            env.current_simulation_step = stop

        out_step_comp = np.array(out_step_comp[:-1])
        out_step = np.array(env.simulation_out)

        sim_is_same = np.allclose(out_step, out_step_comp)

        self.assertTrue(sim_is_same)

