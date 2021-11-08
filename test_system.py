import numpy as np
import control
import matplotlib.pyplot as plt
import time

# define system
pt2_sys = control.tf([1], [0.001, 0.05, 1])  # p = 20; i = 50
pt2_sys_oscillating = control.tf([1], [0.001, 0.005, 1])
sys = pt2_sys
# simulate for 2 sec with 200 Hz
t = np.linspace(0, 2, 20000)
# step at 0.5 sec to 1
step = [0] * 5000 + [1] * 15000
ramp = [0] * 5000 + np.linspace(0, 3, 5000).tolist() + [3] * 10000
u = step
# compute step response for given sys
_, out_pt2 = control.forced_response(sys, t, u)

# add pi controller and compute response with controller
p = 20
i = 50
pi_controller = control.tf([p, i], [1, 0])
open_loop = control.series(pi_controller, sys)
closed_loop = control.feedback(open_loop, 1, -1)
start = time.time()
_, out_pt_controller = control.forced_response(closed_loop, t, u)
print(time.time() - start)


# simulate sys step by step
out_stepped = []
sim_times = []
last_state = None
time_steps_per_sim_step = 100
start = time.time()
for i in range(0, 19999, time_steps_per_sim_step):
    if last_state is not None:
        sim_time, out_step, last_state = control.forced_response(closed_loop, t[i:i + time_steps_per_sim_step+1], u[i:i + time_steps_per_sim_step+1], X0=last_state[:, -1], return_x=True)
    else:
        sim_time, out_step, last_state = control.forced_response(closed_loop, t[i:i+time_steps_per_sim_step+1], u[i:i+time_steps_per_sim_step+1], return_x=True)
    out_stepped = out_stepped + out_step.tolist()[1:]
    sim_times = sim_times + sim_time.tolist()[1:]
out_stepped = np.array(out_stepped)
print(time.time() - start)
# out_stepped = np.reshape(out_stepped, (2000, -1))
plt.plot(t, u, label="Input")
plt.plot(t, out_pt2, label="Open Loop")
plt.plot(t, out_pt_controller, label="Closed Loop")
plt.plot(sim_times, out_stepped, label="Closed Loop Step-by-Step")
plt.grid()
plt.legend()
plt.show()

