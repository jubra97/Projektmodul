import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from sysidentpy.neural_network import NARXNN
from anarx_v2 import ANARX
from narx_miso import NARX

from datatools import load_data
MODEL_NAME  = "MISO_NARX_20_1"

DATASET = "miso.mat"

writer = SummaryWriter(f"runs/{MODEL_NAME}")

def train(net, data, epochs, learning_rate):
    net.learning_rate = learning_rate
    net.epochs = epochs

    u_train, u_valid, u2_train, u2_valid, y_train, y_valid = load_data(data)
    u_train = np.concatenate([u_train, u2_train], axis = 1)
    u_valid = np.concatenate([u_valid, u2_valid], axis = 1)
    train_dl = net.data_transform(u_train, y_train)
    valid_dl = net.data_transform(u_valid, y_valid)

    net.fit(train_dl, valid_dl)
    net.net.global_epochs = net.net.global_epochs + epochs
    writer.add_scalar('Loss/Train', net.train_loss[epochs-1], net.net.global_epochs)
    writer.add_scalar('Loss/Val', net.val_loss[epochs-1], net.net.global_epochs)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    #ax.bar(np.arange(1,21), net.net.weights_by_lag.numpy())
    writer.add_figure('Lags/Weights', fig, net.net.global_epochs)

current_net = pickle.load(open(f"SysIdentPy/models/{MODEL_NAME}.p", "rb" ))
for i in range(50):
    train(current_net, DATASET, 3, 0.00001)
    
if input("save? y/n\n") == "y":
    with open(f"SysIdentPy/models/{MODEL_NAME}.p", "wb") as f:
        pickle.dump(current_net, f)
