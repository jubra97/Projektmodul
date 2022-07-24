import torch
import numpy as np

def lag_matrix(tensor, n_lags):
    t_steps = len(tensor)
    y = torch.zeros((t_steps, n_lags))
    new_tensor = torch.cat((torch.zeros(n_lags), tensor))
    for i in range(t_steps):
        y[i,:] = new_tensor[i+1:i+n_lags+1]
    return y

def normalize(series):
    offset = np.min(series)
    scale = np.max(series)-np.min(series)
    normalized = (series-offset)/scale
    return normalized, scale, offset
    