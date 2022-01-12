from torch import nn
import torch
from sysidentpy.neural_network import NARXNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
import scipy.io
from itertools import groupby

from torch.nn.modules.activation import ReLU

mat = scipy.io.loadmat('SysIdentPy/data/prepr.mat')
u_train = mat['u_train'][0]
u_valid = mat['u_valid'][0]
y_train = mat['y_train'][0]
y_valid = mat['y_valid'][0]




u_train= u_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
u_valid = u_valid.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)

class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(40, 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, 1)
        self.tanh = nn.Tanh()
        self.relu = ReLU()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z

narx_net = NARXNN(
        net=NARX(),
        ylag=20,
        xlag=20,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=2500,
        verbose=True,
        learning_rate=0.00001,
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-08} # optional parameters of the optimizer
)

train_dl = narx_net.data_transform(u_train, y_train)
# for i,elt in enumerate(train_dl):
#     print(elt)
#     if i > 1:
#         break
valid_dl = narx_net.data_transform(u_valid, y_valid)
narx_net.fit(train_dl, valid_dl)
yhat = narx_net.predict(u_valid, y_valid)
ee, ex, extras, lam = narx_net.residuals(u_valid, y_valid, yhat)
narx_net.plot_result(y_valid, yhat, ee, ex, n=2000)

yhat = narx_net.predict(u_train, y_train)
ee, ex, extras, lam = narx_net.residuals(u_train, y_train, yhat)
narx_net.plot_result(y_train, yhat, ee, ex, n=2000)