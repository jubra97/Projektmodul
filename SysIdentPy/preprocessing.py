import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SysIdentPy/data/new_sensor_steps_with_actual_torque')
t = mat['data_2'][0]
t = t[::10]
u = mat['data_2'][2]
u = u[::10]
u = u/max(u)
y = mat['data_2'][3]
y = y[::10]
y = y/max(y)


u_nochnix, u_train, u_valid, u_nix = np.split(u, [140000, 220000, 280000])
u_nochnix, y_train, y_valid, y_nix = np.split(y, [140000, 220000, 280000])
plt.plot(u_valid)
plt.plot(y_valid)
plt.show()
plt.plot(u_train)
plt.plot(y_train)
scipy.io.savemat('newsensorlong.mat', {"u_train": u_train, "u_valid": u_valid, "y_train": y_train, "y_valid": y_valid})

mat = scipy.io.loadmat('newsensorlong.mat')
u_valid = mat['u_valid'][0]
y_valid = mat['y_valid'][0]
plt.plot(u_valid)
plt.plot(y_valid)
#plt.plot(y)

plt.show()