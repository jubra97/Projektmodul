import torch
x = torch.Tensor([  [1,2,3,4],
                    [5,6,7,8]])
list = torch.tensor_split(x, 4, dim=1)
print(torch.cat((list[1], list[3]), 1))
