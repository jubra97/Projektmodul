import pickle

from sysidentpy.neural_network import NARXNN
from anarx_v2 import ANARX

from datatools import load_data

def train(net, data, epochs, learning_rate):
    net.learning_rate = learning_rate
    net.epochs = epochs

    u_train, u_valid, y_train, y_valid = load_data(data)
    train_dl = net.data_transform(u_train, y_train)
    valid_dl = net.data_transform(u_valid, y_valid)

    net.fit(train_dl, valid_dl)
MODEL_NAME  = "ANARX_20_20"
DATASET = "pt2.mat"
current_net = pickle.load(open(f"SysIdentPy/models/{MODEL_NAME}.p", "rb" ))
train(current_net, "pt2.mat", 200, 0.00001)

if input("save? y/n\n") == "y":
    with open(f"SysIdentPy/models/{MODEL_NAME}.p", "wb") as f:
        pickle.dump(current_net, f)