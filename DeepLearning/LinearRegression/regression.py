import random

import numpy as np
import torch
from matplotlib import pyplot as plt

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.001, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    index = list(range(num_examples))
    random.shuffle(index)   # 重排list
    for i in range(0, num_examples, batch_size):
        batch_index = torch.tensor(index[i: min(i + batch_size, num_examples)]) # 当最后不足一个size时  取表尾长度
        yield features[batch_index], labels[batch_index]

def initModel():
    w = torch.normal(0, 1, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    # 回归模型
    return torch.matmul(X, w) + b   # matmul矩阵-向量乘法

def squared_loss(y_hat, y):
    # 均方损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def training():
    # 生成数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 定义参数
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(epoch, ', loss: %.6f' % train_l.mean())

    # plt.scatter(features[:, 1], labels, 1, 'red')
    # x = torch.tensor(np.linspace(-4, 4))
    # y = x * true_w[1] + true_b
    # plt.plot(x, y, 'green')
    # ty = x * float(w[1]) + float(b)
    # plt.plot(x, ty, 'blue')
    # plt.show()
    # print(w[1], '\n', b)


training()
