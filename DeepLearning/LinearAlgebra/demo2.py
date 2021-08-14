import torch

u = torch.tensor([3.0, -4.0])

print(torch.norm(u))    # 计算L2范数（即欧几里得距离）
print(torch.norm(u, 3)) # 计算L3范数
