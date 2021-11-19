from torch import nn
from sysidentpy.neural_network import NARXNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
import scipy.io
from itertools import groupby

mat = scipy.io.loadmat('Messung_Moment_part1.mat')
t = mat['data_1'][0]
u = mat['data_1'][1]
y = mat['data_1'][2]
# Generate a dataset of a simulated dynamical system
x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000,
        colored_noise=False,
sigma=0.001,
train_percentage=80
)
u_nix, u_train, u_valid = np.split(u, [89340, 274500])
u_nix, y_train, y_valid = np.split(y, [89340, 274500])
scale = max(u_train)
u_train=u_train[::200]
u_train=u_train/scale
y_train=y_train[::200]
y_train=y_train/2
y_train=y_train - 0.1
u_valid=u_valid[::200]
u_valid=u_valid/scale
y_valid=y_valid[::200]
y_valid=y_valid/2
y_valid=y_valid - 0.1

u_train= u_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
u_valid = u_valid.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)
class NARX(nn.Module):
        def __init__(self):
                super().__init__()
                self.lin = nn.Linear(55, 100)
                self.lin1 = nn.Linear(100, 500)
                self.lin2 = nn.Linear(500, 100)
                self.lin25 = nn.Linear(100, 10)
                self.lin3 = nn.Linear(10, 1)
                self.tanh = nn.Tanh()

        def forward(self, xb):
                z = self.lin(xb)
                z = self.tanh(z)
                z = self.lin1(z)
                z = self.tanh(z)
                z = self.lin2(z)
                z = self.tanh(z)
                z = self.lin25(z)
                z = self.tanh(z)
                z = self.lin3(z)
                return z

narx_net = NARXNN(
        net=NARX(),
        ylag=50,
        xlag=5,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=5000,
        verbose=True,
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # optional parameters of the optimizer
)

train_dl = narx_net.data_transform(u_train, y_train)
valid_dl = narx_net.data_transform(u_valid, y_valid)
narx_net.fit(train_dl, valid_dl)
print(1)
yhat = narx_net.predict(u_valid, y_valid)
print(2)
ee, ex, extras, lam = narx_net.residuals(u_valid, y_valid, yhat)
print(3)
narx_net.plot_result(y_valid, yhat, ee, ex, n=900)