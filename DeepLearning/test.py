import torch

x = torch.arange(24).reshape(2, 3, 4)
print(x)

print(x.sum(dim=[0, 1]))
