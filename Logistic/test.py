import numpy
from numpy import *

A = mat([[1, 2],
         [3, 4],
         [5, 6]])

label = [0, 1, 2, 3]
K = mat(label).transpose()
P = mat(label)
print(K)
print(P.transpose()) 