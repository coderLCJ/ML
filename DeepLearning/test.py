import torch

x = torch.arange(12)
print(x)

y = torch.zeros_like(x)
y[:] = x
y[0] = -1
print(x, y)