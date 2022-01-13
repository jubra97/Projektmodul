from envs.IOSim import IOSim
import control
import matplotlib.pyplot as plt
import numpy as np

open_loop_sys = control.tf([2], [0.001, 0.05, 1])
open_loop_sys = control.tf2ss(open_loop_sys)

p = 10
i = 10

pi_controller = control.tf2ss(control.tf([p, i], [1, 0]))

io_open_loop = control.LinearIOSystem(open_loop_sys, inputs="u", outputs="y", name="open_loop")
io_pi = control.LinearIOSystem(pi_controller, inputs="e", outputs="u", name="controller")
w_y_comp = control.summing_junction(inputs=["w", "-y_noisy"], output="e")
y_noise = control.summing_junction(inputs=["y", "noise"], outputs="y_noisy")

closed_loop = control.interconnect([w_y_comp, io_pi, io_open_loop, y_noise], name="closed_loop",
                                   inplist=["w", "noise"],
                                   outlist=["y", "y_noisy", "u", "e"])

print(closed_loop)

sim = IOSim(closed_loop, 10_000, 200, 100, 1, 1)
u = np.array([[0] * 5000 + [1] * 10_000, [0] * 15_000])

states = np.array([]).reshape((3, 0))
for i in range(150):
    sim.sim_one_step(u[:, i*100:(i+1)*100+1])
    states = np.concatenate((states, sim.last_state), axis=1)

plt.plot(sim._sim_out[0, :])
plt.plot(sim._sim_out[2, :])
print(np.array(u)[0, :])
plt.plot(np.array(u)[0, :].T)
plt.plot(states.T)
plt.show()
