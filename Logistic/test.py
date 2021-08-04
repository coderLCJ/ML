import numpy
from numpy import *

L = list(range(10))
del L[1]
print(L)
L.remove(L[8])
print(L)