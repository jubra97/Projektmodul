import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('messung_part1.mat')
t = mat['data_1'][0]
t = t[::200]
u = mat['data_1'][1]
u = u/max(u)
u = u[::200]
y = mat['data_1'][2]
y = y/max(y)
y = y[::200]

u_nochnix, u_train, u_valid, u_nix = np.split(u, [500, 2500, 4500])
u_nochnix, y_train, y_valid, y_nix = np.split(y, [500, 2500, 4500])
plt.plot(u_valid)
plt.plot(y_valid)
#plt.plot(y)

plt.show()
scipy.io.savemat('prepr.mat', {"u_train": u_train, "u_valid": u_valid, "y_train": y_train, "y_valid": y_valid})

mat = scipy.io.loadmat('prepr.mat')
u_valid = mat['u_valid'][0]
y_valid = mat['y_valid'][0]
plt.plot(u_valid)
plt.plot(y_valid)
#plt.plot(y)

plt.show()