from mxnet import nd

# ----------------------------------------------------- #
# 基本操作
# ----------------------------------------------------- #
X = nd.arange(12)
print(X)
print(X.shape)
print(X.size)

X = X.reshape((3, 4))
print(X)

X = nd.random.normal(0, 1, shape=(3, 4))
print(X)
Y = nd.random.normal(0, 1, shape=(3, 4))
print(Y)

X*Y         # 元素相乘
nd.dot(X, Y.T)    # 矩阵乘法

X = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
Y = nd.array([[2, 4, 6, 8], [3, 6, 7, 9], [0, 2, 4, 8]])
print(nd.concat(X, Y, dim=0))   # 维度为0 即在行上连接
print(nd.concat(X, Y, dim=1))   # 维度为1 即在列上连接

print(X == Y)   # 生成一个判断结果矩阵 0代表不同 1代表相同

T = X.sum()
print(T)                    # 求和 得到一个元素的NDArray
print(T.norm().asscalar())  # 转换成标量