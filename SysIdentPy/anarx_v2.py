from torch import nn
import torch
from sysidentpy.neural_network import NARXNN
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
import pickle

from torch.nn.modules.activation import ReLU


class ANARX(nn.Module):
    def __init__(self, output_lags, input_lags):
        super().__init__()
        self.global_epochs = 0
        self.output_lags = output_lags
        self.input_lags = input_lags
        self.lin_io1 = nn.ModuleList([nn.Linear(2, 5) for i in range(min(output_lags,input_lags))])
        self.lin_io2 = nn.ModuleList([nn.Linear(5, 5) for i in range(min(output_lags,input_lags))])
        self.lin_io3 = nn.ModuleList([nn.Linear(5, 1) for i in range(min(output_lags,input_lags))])
        self.lin1 = nn.ModuleList([nn.Linear(1, 5) for i in range(max(output_lags,input_lags)-min(output_lags,input_lags))])
        self.lin2 = nn.ModuleList([nn.Linear(5, 5) for i in range(max(output_lags,input_lags)-min(output_lags,input_lags))])
        self.lin3 = nn.ModuleList([nn.Linear(5, 1) for i in range(max(output_lags,input_lags)-min(output_lags,input_lags))])
        self.activ2 = nn.Tanh()
        self.activ1 = nn.Tanh()
        self.weights_by_lag_batch = torch.empty(100, (max(self.input_lags, self.output_lags)))
        self.weights_by_lag = torch.empty(max(self.input_lags, self.output_lags))
    def forward(self, xb):
        x_list = torch.tensor_split(xb, self.output_lags+self.input_lags, dim=1)
        z = 0
        self.weights_by_lag_batch = torch.empty(xb.shape[0], (max(self.input_lags, self.output_lags)))
        for i in range(min(self.output_lags,self.input_lags)):
            input_i = x_list[self.output_lags+i]
            output_i = x_list[i]
            y = torch.cat((output_i, input_i), dim = 1)
            y = self.lin_io1[i](y)
            y = self.activ1(y)
            y = self.lin_io2[i](y)
            y = self.activ2(y)
            y = self.lin_io3[i](y)
            self.weights_by_lag_batch[:, i] = torch.squeeze(y)
            z = z + y
        for i in range(max(self.output_lags,self.input_lags)-min(self.output_lags,self.input_lags)):
            y = x_list[i]
            y = self.lin1[i](y)
            y = self.activ1(y)
            y = self.lin2[i](y)
            y = self.activ2(y)
            y = self.lin3[i](y)
            self.weights_by_lag_batch[:, i+min(self.output_lags,self.input_lags)] = torch.squeeze(y)
            z = z + y
        self.weights_by_lag_batch = self.weights_by_lag_batch/self.weights_by_lag_batch.sum(axis = 1)[:,None]
        self.weights_by_lag = torch.mean(self.weights_by_lag_batch, 0)
        return z



def build_net(output_lags, input_lags):
    narx_net = NARXNN(
        net=ANARX(output_lags, input_lags),
        ylag=output_lags,
        xlag=input_lags,
        loss_func='mse_loss',
        optimizer='Adam',
        epochs=100,
        verbose=True,
        learning_rate=0.0001,
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-08} # optional parameters of the optimizer
    )
    with open(f"SysIdentPy/models/ANARX_{output_lags}_{input_lags}.p", "wb") as f:
        pickle.dump(narx_net, f)


if __name__ == "__main__":
    build_net(20, 20)

    

