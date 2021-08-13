import torch

# ----------------------------------------------------- #
# 广播机制
# ----------------------------------------------------- #
A = torch.arange(3).reshape((3, 1))
B = torch.arange(2).reshape((1, 2))
'''
A = 
[[0.]
 [1.]
 [2.]] 

B =
[[0. 1.]]
'''

C = A + B
'''
广播机制：
A = 
[[0. 0.]
 [1. 1.]
 [2. 2.]] 

B =
[[0. 1.]
 [0. 1.]
 [0. 1.]]
'''
print(C)