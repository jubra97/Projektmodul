# import numpy as np
# import matplotlib.pyplot as plt
# from envs.PIAdaptivePT2 import PIAdaptivePT2
#
# env = PIAdaptivePT2(log=True)
#
# env.reset(1, 0)
# while not env.sim.done:
#     env.step([-0.99])
#
# # plt.plot(env.sim.t, env.sim.sensor_out)
# plt.plot(env.sim.t, env.sim._sim_out)
# plt.plot(env.sim.t, env.sim.e)
# plt.grid()
# plt.show()


import control
import numpy as np
import matplotlib.pyplot as plt
from envs.TfSimTest import TfSim

controller_p = 2
controller_i = 0

plant = control.tf2ss(control.tf([1], [0.001, 0.05, 1]))
pi_controller = control.tf2ss(control.tf([controller_p, controller_i], [1, 0]))
t = np.linspace(1e-200, 1.5, 15000)
print(t[0:3])
u = [0] * 5000 + [1] * 10000
noise = np.random.normal(0, 0.00, 15000)

inputs = np.array([u, noise])
print(inputs.shape)

io_open_loop = control.LinearIOSystem(plant, inputs="u", outputs="y", name="open_loop")
io_pi = control.LinearIOSystem(pi_controller, inputs="e", outputs="u", name="controller")
w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

closed_loop = control.interconnect([w_y_comp, io_pi, io_open_loop, y_noise], name="closed_loop",
                                   inplist=["w", "noise"],
                                   outlist=["y", "y_noisy", "u", "e"])

t, y = control.input_output_response(closed_loop, t[:201], inputs[:, :201])
print("Geht")
sim = TfSim(closed_loop, 10_000, 200, 200, 1.5)
while sim.done is False:
    sim.sim_one_step(inputs[:, sim.current_simulation_step:sim.current_simulation_step+sim.model_steps_per_controller_update+1], add_noise=False)
# t, y = control.input_output_response(closed_loop, t, inputs)

# plt.plot(sim.t, sim._sim_out[0], label="y")
# plt.plot(sim.t, sim._sim_out[1], label="y_noisy")
# plt.plot(sim.t, sim._sim_out[2], label="u")
# plt.plot(sim.t, sim._sim_out[3], label="e")
# plt.grid()
# plt.legend()
#
# plt.show()

print(io_open_loop)
print(io_pi)
print(closed_loop)