import time
import torch

device = torch.device('cpu')
s = time.time()
x = torch.normal(0, 1, (1000, 1000), device=device)
y = torch.normal(0, 1, (1000, 1000), device=device)
for i in range(100000):
    x = x * y
print(x)
x = x * y
e = time.time()
print(e - s)
