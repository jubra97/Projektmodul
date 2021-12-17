import scipy.io
import numpy as np
import pandas as pd
import control
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('messung_part1.mat')
u = mat['data_1'][1]
u = u/max(u)
u = u[::200]
u_nochnix, u_train, u_valid, u_nix = np.split(u, [500, 2500, 4500])


T = 0.01
D= 1.5
sys = control.tf([1], [T**2, D*T, 1])
t_train = np.linspace(0, 3.5, len(u_train))
t_valid = np.linspace(0, 3.5, len(u_valid))
_,out_train = control.forced_response(sys, t_train, u_train)
_,out_valid = control.forced_response(sys, t_valid, u_valid)
maxi = max(max(out_train), max(out_valid))
out_train = out_train/maxi
out_valid = out_valid/maxi

plt.plot(out_train)
plt.plot(out_valid)
plt.show()

scipy.io.savemat('pt2.mat', {"u_train": u_train, "u_valid": u_valid, "y_train": out_train, "y_valid": out_valid})
