from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 求出每一列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # minVals 和 ranges 的大小都为 1*3
    normDataSet = dataSet - tile(minVals, (m, 1)) # 原始数据的每一行都减去所在列的最小值
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 读取测试数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

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
    # numpy.argsort() 函数返回的是数组值从小到大的索引值
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

# 测试函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) # 取10%的数据用于测试
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classfiy0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['完全不感兴趣', '有点喜欢', '很喜欢']
    percentTats = float(input('打游戏的时间比例：'))
    ffMiles = float(input('每年飞行公里数：'))
    iceCream = float(input('每年消耗的冰淇淋公升数：'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    # 需要将输入的测试数据特征归一化
    res = classfiy0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('你可能对这个人: ', resultList[res-1])

classifyPerson()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(mat[:, 0], mat[:, 1], s=5.0*array(labels), c=5.0*array(labels))
# plt.show()
'''
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
参数说明：
x，y：长度相同的数组，也就是我们即将绘制散点图的数据点，输入数据。
s：点的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小。
c：点的颜色，默认蓝色 'b'，也可以是个 RGB 或 RGBA 二维行数组。
marker：点的样式，默认小圆圈 'o'。
cmap：Colormap，默认 None，标量或者是一个 colormap 的名字，只有 c 是一个浮点数数组的时才使用。如果没有申明就是 image.cmap。
norm：Normalize，默认 None，数据亮度在 0-1 之间，只有 c 是一个浮点数的数组的时才使用。
vmin，vmax：：亮度设置，在 norm 参数存在时会忽略。
alpha：：透明度设置，0-1 之间，默认 None，即不透明。
linewidths：：标记点的长度。
edgecolors：：颜色或颜色序列，默认为 'face'，可选值有 'face', 'none', None。
plotnonfinite：：布尔值，设置是否使用非限定的 c ( inf, -inf 或 nan) 绘制点。
**kwargs：：其他参数。
以下实例 scatter() 函数接收长度相同的数组参数，一个用于 x 轴的值，另一个用于 y 轴上的值：
'''

