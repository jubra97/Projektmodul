import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('SysIdentPy/data/messung_part1neuneu')
t = mat['data_1'][0]
t = t[::10]
u = mat['data_1'][1]
u = u[::10]
u = u - min(u)
u = u/max(u)
u2 = mat['data_1'][4]
u2 = u2[::10]
u2 = u2 - min(u2)
u2 = u2/max(u2)
y = mat['data_1'][3]
y = y[::10]
y = y - min(y)
y = y/max(y)


u_nochnix, u_train, u_valid, u_nix = np.split(u, [140000, 220000, 300000])
u2_nochnix, u2_train, u2_valid, u2_nix = np.split(u2, [140000, 220000, 300000])
u_nochnix, y_train, y_valid, y_nix = np.split(y, [140000, 220000, 300000])
plt.plot(u_valid)
plt.plot(y_valid)
plt.plot(u2_valid)
plt.show()
plt.plot(u_train)
plt.plot(y_train)
scipy.io.savemat('miso.mat', {"u_train": u_train, "u_valid": u_valid, "u2_train": u2_train, "u2_valid": u2_valid, "y_train": y_train, "y_valid": y_valid})

mat = scipy.io.loadmat('miso.mat')
u_valid = mat['u_valid'][0]
y_valid = mat['y_valid'][0]
plt.plot(u_valid)
plt.plot(y_valid)
#plt.plot(y)

plt.show()