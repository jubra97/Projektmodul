from turtle import forward
import torch
import torch.nn as nn
from ANARX import ANARX
import numpy as np

class ANARX_SS(nn.Module):
    def __init__(self, reference_model: ANARX):
        super(ANARX_SS, self).__init__()

        self.n_subnets = reference_model.n_subnets
        self.subnets = reference_model.subnets
        self.lag_map = reference_model.lag_map
        self.shift_matrix = torch.Tensor(np.eye(self.n_subnets, k=1)).detach()
    
    def forward(self, state: torch.Tensor, y: torch.Tensor, u: torch.Tensor):
        shifted_state = torch.matmul(self.shift_matrix, state)
        subnetoutputs = torch.zeros(self.n_subnets)
        for i in range(self.n_subnets):
            if self.lag_map[i][0] == 1:
                input = torch.cat((u, y))
            else:
                input = y
            subnetoutputs[i] = self.subnets[i](input)
        next_state = shifted_state + subnetoutputs
        return next_state


