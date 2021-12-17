from torch import nn
import torch
from sysidentpy.neural_network import NARXNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
import scipy.io

from torch.nn.modules.activation import ReLU

mat = scipy.io.loadmat('prepr.mat')
u_train = mat['u_train'][0]
u_valid = mat['u_valid'][0]
y_train = mat['y_train'][0]
y_valid = mat['y_valid'][0]


OUTPUT_LAGS = 5
INPUT_LAGS = 1

u_train= u_train.reshape(-1,1)
y_train= y_train.reshape(-1,1)
u_valid = u_valid.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)

class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_io1 = nn.ModuleList([nn.Linear(2, 5) for i in range(min(OUTPUT_LAGS,INPUT_LAGS))])
        self.lin_io2 = nn.ModuleList([nn.Linear(5, 5) for i in range(min(OUTPUT_LAGS,INPUT_LAGS))])
        self.lin_io3 = nn.ModuleList([nn.Linear(5, 1) for i in range(min(OUTPUT_LAGS,INPUT_LAGS))])
        self.lin1 = nn.ModuleList([nn.Linear(1, 5) for i in range(max(OUTPUT_LAGS,INPUT_LAGS)-min(OUTPUT_LAGS,INPUT_LAGS))])
        self.lin2 = nn.ModuleList([nn.Linear(5, 5) for i in range(max(OUTPUT_LAGS,INPUT_LAGS)-min(OUTPUT_LAGS,INPUT_LAGS))])
        self.lin3 = nn.ModuleList([nn.Linear(5, 1) for i in range(max(OUTPUT_LAGS,INPUT_LAGS)-min(OUTPUT_LAGS,INPUT_LAGS))])
        self.activ2 = nn.Tanh()
        self.activ1 = nn.Tanh()
    def forward(self, xb):
        x_list = torch.tensor_split(xb, OUTPUT_LAGS+INPUT_LAGS, dim=-1)
        z = 0
        for i in range(min(OUTPUT_LAGS,INPUT_LAGS)):
            y = torch.cat((x_list[i], x_list[min(OUTPUT_LAGS,INPUT_LAGS)+i]), dim = 1)
            y = self.lin_io1[i](y)
            y = self.activ1(y)
            y = self.lin_io2[i](y)
            y = self.activ2(y)
            y = self.lin_io3[i](y)
            z = z + y
        for i in range(max(OUTPUT_LAGS,INPUT_LAGS)-min(OUTPUT_LAGS,INPUT_LAGS)):
            y = x_list[i]
            y = self.lin1[i](y)
            y = self.activ1(y)
            y = self.lin2[i](y)
            y = self.activ2(y)
            y = self.lin3[i](y)
            z = z + y
        return z

narx_net = NARXNN(
        net=NARX(),
        ylag=OUTPUT_LAGS,
        xlag=INPUT_LAGS,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=1000,
        verbose=True,
        learning_rate=0.0001,
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