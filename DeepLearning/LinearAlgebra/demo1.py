import torch

x, y = torch.arange(4, dtype=torch.float32), torch.ones(4)
print(x, y)

print(torch.dot(x, y))  # 计算点积
print(torch.sum(x*y))   # 计算乘积后累加 等于点积的值

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.arange(4, dtype=torch.float32)
print(A, B)

print(torch.mv(A, B))   # 计算矩阵-向量积

B = torch.ones((4, 1))
print(torch.mm(A, B))   # 计算矩阵-矩阵乘法

B = torch.ones((5, 4))
print(A*B)              # 计算哈达玛积（对应元素相乘）
