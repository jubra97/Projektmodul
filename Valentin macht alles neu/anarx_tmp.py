import torch
import torch.nn as nn


class subnet(nn.Module):
    def __init__(self, inputs):
        super(subnet, self).__init__()
        self.fc1 = nn.Linear(inputs, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


class ANARX(nn.Module):
    def __init__(self, lags):
        super(ANARX, self).__init__()
        self.nlags = lags
        self.subnets = nn.ModuleList([subnet(2) for i in range(lags)])

    def forward(self, x_lagged, y_lagged):
        x = torch.empty((self.nlags))
        # Compute Subnet Outputs
        for i, subnet in enumerate(self.subnets):
            output = subnet(torch.stack((x_lagged[i], y_lagged[i])))
            x[i] = output
        return torch.sum(x)

    def initLags(self):
        return torch.zeros((self.nlags))


class ANARX_TWO(nn.Module):
    def __init__(self, lags):
        super(ANARX_TWO, self).__init__()
        self.nlags = lags
        self.subnets = nn.ModuleList([subnet(3) for i in range(lags)])

    def forward(self, x1_lagged, x2_lagged, y_lagged):
        x = torch.empty((self.nlags))
        # Compute Subnet Outputs
        for i, subnet in enumerate(self.subnets):
            output = subnet(torch.stack(
                (x1_lagged[i], x2_lagged[i], y_lagged[i])))
            x[i] = output
        return torch.sum(x)

    def initLags(self):
        return torch.zeros((self.nlags))


class subnet_v2(nn.Module):
    def __init__(self, inputs):
        super(subnet, self).__init__()
        self.fc1 = nn.Linear(inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ANARX_TWO_v2(nn.Module):
    def __init__(self, lags):
        super(ANARX_TWO, self).__init__()
        self.nlags = lags
        self.subnets = nn.ModuleList([subnet(3) for i in range(lags)])

    def forward(self, x1_lagged, x2_lagged, y_lagged):
        x = torch.empty((self.nlags))
        # Compute Subnet Outputs
        for i, subnet in enumerate(self.subnets):
            output = subnet(torch.stack(
                (x1_lagged[i], x2_lagged[i], y_lagged[i])))
            x[i] = output
        return torch.sum(x)

    def initLags(self):
        return torch.zeros((self.nlags))