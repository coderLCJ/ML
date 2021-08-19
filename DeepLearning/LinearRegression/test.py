import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils import data

a = torch.arange(10)
b = torch.arange(10, 20)
print(a, b)
c = data.TensorDataset(a)
for i in c:
    print(i)