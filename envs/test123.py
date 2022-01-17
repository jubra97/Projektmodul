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
                                   inplist=["w", "noise"], inputs=["w", "noise"],
                                   outlist=["y", "y_noisy", "u", "e"])

print(closed_loop)

sim = IOSim(closed_loop, 10_000, 200, 100, 1, 1)
u = np.array([[0.2] * 5000 + [0.5] * 10_000, [0] * 15_000])

T = control.timeresp._default_time_vector(open_loop_sys)
U = np.ones_like(T) * u[0,0]/2
_, step_response, states = control.forced_response(open_loop_sys, T, U, return_x=True)
last_state = np.array([states[:, -1]]).T
sim.last_state = np.concatenate((np.array([[0]]), last_state))

sim.last_state[0, 0] = u[0,0] / (i*2)

a = (sim.last_state[:, -1] @ sim.sys.C.T)[2]
print(a)
print(last_state)
print(sim.last_state)

states = np.array([]).reshape((3, 0))
for i in range(150):
    sim.sim_one_step(u[:, i*100:(i+1)*100+1])
    states = np.concatenate((states, sim.last_state), axis=1)


fig, ax = plt.subplots(1, 2)
ax[0].plot(sim._sim_out[0, :], label="y")
ax[0].plot(sim._sim_out[2, :], label="u")
print(np.array(u)[0, :])
ax[0].plot(np.array(u)[0, :].T, label="w")
ax[0].grid()
ax[0].legend()
ax[1].plot(states.T[:, 0], label="controller_x0")
ax[1].plot(states.T[:, 1], label="open_loop_x0")
ax[1].plot(states.T[:, 2], label="open_loop_x1")
ax[1].grid()
ax[1].legend()
plt.show()
