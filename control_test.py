import control
import matplotlib.pyplot as plt
import numpy as np
p = 0.12
i = 10

sys = control.tf([5.9e11, 1.2e12], [1, 255, 1e5, 1.6e7, 2.3e9, 1.7e11, 3.7e11])
pi_controller = control.tf([p, i], [1, 0])
open_loop = control.series(pi_controller, sys)

closed_loop = control.feedback(open_loop, 1, -1)
print(closed_loop)


#closed_loop = control.series(1, closed_loop)

print(closed_loop)

t = np.linspace(0, 3.5, 10000)
# out = control.forced_response(sys, t, 1)
# plt.plot(out[0], out[1])

out = control.forced_response(closed_loop, t, 1)
plt.plot(out[0], out[1])
plt.grid()



plt.show()