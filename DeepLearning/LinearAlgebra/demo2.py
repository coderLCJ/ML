import torch

u = torch.tensor([3.0, -4.0])

print(torch.norm(u))    # 计算L2范数（即欧几里得距离）
print(torch.norm(u, 3)) # 计算L3范数

m = torch.ones(4, 9)
print(m)
print(torch.norm(m))    # 菲罗贝尼乌斯范数：矩阵元素的平方和的平方根
