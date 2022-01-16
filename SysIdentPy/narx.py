from torch import nn
import torch
from sysidentpy.neural_network import NARXNN
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
import pickle

from torch.nn.modules.activation import ReLU



class NARX(nn.Module):
    def __init__(self, output_lags, input_lags):
        super().__init__()
        self.global_epochs = 0
        self.output_lags = output_lags
        self.input_lags = input_lags
        self.lin = nn.Linear(self.output_lags+self.input_lags, 60)
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

def build_net(output_lags, input_lags):
    narx_net = NARXNN(
        net=NARX(output_lags, input_lags),
        ylag=output_lags,
        xlag=input_lags,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=100,
        verbose=True,
        learning_rate=0.0001,
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-08} # optional parameters of the optimizer
    )
    with open(f"SysIdentPy/models/NARX_{output_lags}_{input_lags}.p", "wb") as f:
        pickle.dump(narx_net, f)


if __name__ == "__main__":
    build_net(6, 1)