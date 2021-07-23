from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for labels in dataSet:
        lable = labels[-1]
        if lable not in labelCounts.keys():
            labelCounts[lable] = 0
        labelCounts[lable] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2) # log以2为底
    return shannonEnt

def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'no'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeature(dataSet):
    featureLen = len(dataSet[0]) - 1
    baseEnt = calcShannonEnt(dataSet)
    bestFeature = -1
    bestInfo = 0
    for i in range(featureLen):
        Val = [eg[i] for eg in dataSet]
        featureVal = set(Val)
        info = 0
        for j in featureVal:
            spDataSet = splitDataSet(dataSet, i, j)
            prob = len(spDataSet)/float(len(dataSet))
            info += prob * calcShannonEnt(spDataSet)
        addinfo = baseEnt - info
        if addinfo > bestInfo:
            bestInfo = addinfo
            bestFeature = i
    return bestFeature

def majoritCnt(classList):
    classClount = {}
    for vote in classList:
        if vote not in classClount.keys():
            classClount[vote] = 0
        classClount[vote] += 1
    sortedClassCount = sorted(classClount.items(), key=operator.itemgetter(1), reverse=True)    # 返回的是一个列表
    return sortedClassCount[0][0]

def createTree(dataSet, Labels):
    labels = Labels[:]
    classList = [eg[-1] for eg in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:    # 使用完所有特征只剩类别时 返回类别出现次数最多的分类
        return majoritCnt(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])   # 删除该特征
    featVal = [fg[bestFeat] for fg in dataSet]
    uniqueFeat = set(featVal)
    for vals in uniqueFeat:
        subLabels = labels[:]   # 复制标签
        myTree[bestFeatLabel][vals] = createTree(splitDataSet(dataSet, bestFeat, vals), subLabels)  # 以该值划分数据集
    return myTree



'''
测试不同划分下的数据集的有序度
print(splitDataSet(dataSet, 0, 0))
print(splitDataSet(dataSet, 0, 1))
print('-------------------------')
print(splitDataSet(dataSet, 1, 0))
print(splitDataSet(dataSet, 1, 1))
'''

