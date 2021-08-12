import torch

# ----------------------------------------------------- #
# 基本操作
# ----------------------------------------------------- #
X = torch.arange(12)
print(X)
print(X.shape)   # 张量形状
print(X.numel()) # 元素个数

X = X.reshape((3, 4))
print(X)

X = torch.randn(3, 4)
print(X)

x = torch.tensor([[1, 2, 4, 8]])
y = torch.tensor([[2, 2, 2, 2]])
print(x)
print(y)


print(torch.cat((x, y), dim=0)) # 按行连接
print(torch.cat((x, y), dim=1)) # 按列连接

