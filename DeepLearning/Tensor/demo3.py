import torch
import numpy as np

# ----------------------------------------------------- #
# Tensor 和 NumPy 互换
# ----------------------------------------------------- #
X = torch.arange(1)
A = X.numpy()
B = torch.tensor(A)

print(type(A))
print(type(B))

print(A.item()) # 转换成标量
print(float(A))

