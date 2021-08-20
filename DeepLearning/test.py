import math
import os

import numpy as np
import sympy
import torch
from matplotlib import pyplot as plt


x = torch.normal(0, 1, size=(3, 3), dtype=torch.float32)
y = torch.tensor([0.], dtype=torch.float32)
print(x)
y = x.argmax(axis=0)
print(y)
