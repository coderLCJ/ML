from numpy import *
import operator

def createDataSet():
    group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']
    return group, labels

def classfiy0(inX, dataSet, labels, k):
    # ndarray.shape 表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。比如，一个二维数组，其维度表示"行数"和"列数"。
    dataSetSize = dataSet.shape[0]
    # tile 重复b的各个维度 tile(b, (x,y))
    diffmat = tile(inX, (dataSetSize, 1)) - dataSet # 可直接写成 inX - dataSet
    sqDiffMat = diffmat**2      # 数组每个元素平方
    # 按行相加，并且保持其二维特性 keepdims=True 不写则默认不保持
    # axis: 0 按列 1 按行
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # numpy.argsort() 函数返回的是数组值从小到大的索引值。
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # get(key, default=None) default: 不存在时返回该值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 按第二个关键字 即出现的次数排序 reverse=True即从大到小
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的元素的标签
    return sortedClassCount[0][0]

group, labels = createDataSet()
print(classfiy0([60, 60], group, labels, 3))

