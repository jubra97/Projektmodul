import control
import matplotlib.pyplot as plt
import numpy as np
p = 0.1
i = 9
T = 0.15
D= 1.5
sys = control.tf([1], [T**2, D*T, 1])
pi_controller = control.tf([p, i], [1, 0])
open_loop = control.series(pi_controller, sys)
u = [0]*3000 + [1] * 7000
closed_loop = control.feedback(open_loop, 1, -1)

#closed_loop = control.series(1, closed_loop)


t = np.linspace(0, 3.5, 10000)
# out = control.forced_response(sys, t, 1)
# plt.plot(out[0], out[1])

out = control.forced_response(sys, t, u)
reward = (np.square(out[1] - u)).mean()
print(reward)
plt.plot(out[0], out[1])
plt.grid()



plt.show()