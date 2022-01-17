import numpy
import torch
import matplotlib.pyplot as plt
x = torch.Tensor([  [2,2,3,4],
                    [6,6,7,8]])
list = torch.tensor_split(x, 4, dim=1)
print(torch.cat((list[1], list[3]), 1))
w = x/x.sum(axis = 1)[:,None]
w = torch.mean(w, 0)
print(numpy.arange(1,21))
