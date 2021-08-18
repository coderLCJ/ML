import regression
import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt




true_w = torch.tensor([2, -3.4])
true_b = 4.2
W = torch.tensor([[ 0.0164],
        [-0.0006]])

X = torch.tensor([[ 0.6119, -1.6236],
        [ 1.9420, -0.2645],
        [ 0.9907, -1.1116],
        [-1.3452,  0.7416],
        [-1.3630,  0.9793],
        [-0.0662,  0.7524],
        [ 0.0197,  0.5318],
        [-0.0379, -0.6951],
        [ 2.7327,  0.3512],
        [ 0.5026,  0.5711]])

y = torch.matmul(X, true_w) + true_b
y_hat = torch.matmul(X, W)
print((y-y_hat).sum()/10)