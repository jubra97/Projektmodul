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
import time

import control
import numpy as np
import matplotlib.pyplot as plt
from envs.IOSim import IOSim

controller_p = 100
controller_i = 0

plant = control.tf2ss(control.tf([1], [0.001, 0.05, 1]))
pi_controller = control.tf2ss(control.tf([controller_p, controller_i], [1, 0]))
t = np.linspace(0, 1.5, 15000)
a = np.nextafter(0, 1)
print(a)
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
start = time.perf_counter()
t, y = control.input_output_response(closed_loop, t, inputs)
print(time.perf_counter() - start)


sim = IOSim(closed_loop, 10_000, 200, 200, 1.5)


start = time.perf_counter()
while sim.done is False:
    sim.sim_one_step(inputs[:, sim.current_simulation_step:sim.current_simulation_step+sim.model_steps_per_controller_update+1])
print(time.perf_counter() - start)
# t, y = control.input_output_response(closed_loop, t, inputs)
print(sim.t_sensor)
print(sim.t_sensor.shape)
# plt.plot(sim.t, sim._sim_out[0], label="y_stepped")
# plt.plot(sim.t, sim._sim_out[1], label="y_noisy_stepped")
# plt.plot(sim.t, sim._sim_out[2], label="u_stepped")
# plt.plot(sim.t, sim._sim_out[3], label="e_stepped")

plt.plot(abs(np.diff(sim.sensor_out[2])))

# plt.plot(sim.t, y[0], label="y")
# plt.plot(sim.t, y[1], label="y_noisy")
# plt.plot(sim.t, y[2], label="u")
# plt.plot(sim.t, y[3], label="e")

plt.grid()
plt.legend()

plt.show()

# print(np.isclose(t, sim.t).all())
# print(np.isclose(y, sim._sim_out).all())
#
# print(io_open_loop)
# print(io_pi)
# print(closed_loop)