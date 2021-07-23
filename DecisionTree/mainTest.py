from trees import *

def myClassify(inputTree, featLables, testVec):
    if type(inputTree).__name__ == 'str':
        return inputTree
    for key in inputTree.keys():
        val = set(inputTree[key])
        if testVec[0] in val:
            temp = testVec[0]
            del testVec[0]
            return myClassify(inputTree[key][temp], featLables, testVec)
        else:
            return 'no class'

def classify(inputTree, featLables, testVec):
    global classLabel
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)  # 将特征转化为下标
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel

# 存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle # 默认二进制操作文件
    fr = open(filename, 'rb')
    return pickle.load(fr)


# dataSet, labels = creatDataSet()
# tree = createTree(dataSet, labels)
# print(tree)
# storeTree(tree, 'Tree.txt')
# print(grabTree('Tree.txt'))