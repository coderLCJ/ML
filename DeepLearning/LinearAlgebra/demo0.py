import torch

A = torch.arange(20, dtype=torch.float32).reshape((5, 4))
print(A)

print(A.sum())  # 总体求和

print(A.sum(dim=0)) # 按轴0降维求和（按行）
print(A.sum(dim=1)) # 按轴1降维求和（按列）
print(A.sum(dim=[0, 1]))    # 等价与A.sum()

print(A.mean()) #总体求均值
print(A.mean(dim=0))    # 按行累加求均值
print(A.mean(dim=1))    # 按列累加求均值

print(A.sum(dim=1, keepdim=True))   # 保持维度
print(A.cumsum(dim=0))              # 计算某个轴元素的累积总和