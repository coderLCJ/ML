import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x, x)
y.backward()        # 调用反向传播函数计算梯度
print(x.grad)       # 输出梯度
print(x.grad == 4*x)

x.grad.zero_()  # pytorch会累计梯度 需要手动清除
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()  # 把y视作常数
z = u * x       # 求导则是 u 即 x^2
z.sum().backward()
print(x.grad)

