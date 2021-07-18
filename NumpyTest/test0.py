from numpy import *

randMat = mat(random.rand(4, 4))    # mat 将数组转化为矩阵
irandMat = randMat.I    # 求逆矩阵
E = eye(4)                # 4*4的单位矩阵
print(E)