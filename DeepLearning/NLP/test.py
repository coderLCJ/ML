# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         test
# Description:  
# Author:       Laity
# Date:         2021/9/12
# ---------------------------------------------
import torch
import numpy as np


x = torch.ones(3, 3, requires_grad=True)
print(x)
y = 2 * x + 1
y.mean().backward()
print(x.grad)

y = torch.randn(3, 5)
print(y)
print(y.size())
