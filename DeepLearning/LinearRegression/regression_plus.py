import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def load_array(data_arrays, batch_size, is_train=True):
    """ 数据迭代器 """
    dataSet = data.TensorDataset(*data_arrays)  # *代表解开list入参 TensorDataset对tensor进行打包 一一对应 在下面进行打乱时就不会混乱
    return data.DataLoader(dataSet, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))    # (2,1) 输入形状和输出形状
net[0].weight.data.normal_(0, 0.001)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()     # 定义损失函数，平方L2范数 返回样本损失的平均值 此步已经求均值
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

epochs = 3
for epoch in range(epochs):
    for X, y in data_iter:
        l = loss(net(X), y) # net(X) 预测值 y 真实值
        trainer.zero_grad()
        l.backward()    # loss已求sum 可以直接求导
        trainer.step()  # 更新梯度
    l = loss(net(features), labels)
    print(epoch, 'loss = %.6f' % l)
