import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


x = torch.arange(10, requires_grad=True, dtype=torch.float32)
y = x ** 2
print(x)
y.backward()
print(x.grad)

