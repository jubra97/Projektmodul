import pickle
from torch import nn
import torch
from anarx_v2 import ANARX
from sysidentpy.neural_network import NARXNN
import scipy.io
from datatools import load_data

u_train, u_valid, u2_train, u2_valid, y_train, y_valid = load_data("miso.mat")

net = pickle.load(open("SysIdentPy/models/MISO_NARX_20_1.p", "rb" ))

yhat = net.predict(u_train, y_train)
ee, ex, extras, lam = net.residuals(u_train, y_train, yhat)
net.plot_result(y_train, yhat, ee, ex, n=60000)

yhat = net.predict(u_valid, y_valid)
ee, ex, extras, lam = net.residuals(u_valid, y_valid, yhat)
net.plot_result(y_valid, yhat, ee, ex, n=60000)