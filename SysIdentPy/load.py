import pickle
from torch import nn
import torch
from anarx_v2 import ANARX
from sysidentpy.neural_network import NARXNN
import scipy.io
from datatools import load_data

u_train, u_valid, y_train, y_valid = load_data("pt2.mat")

net = pickle.load(open("SysIdentPy/models/ANARX_20_20.p", "rb" ))

yhat = net.predict(u_train, y_train)
ee, ex, extras, lam = net.residuals(u_train, y_train, yhat)
net.plot_result(y_train, yhat, ee, ex, n=2000)

yhat = net.predict(u_valid, y_valid)
ee, ex, extras, lam = net.residuals(u_valid, y_valid, yhat)
net.plot_result(y_valid, yhat, ee, ex, n=2000)