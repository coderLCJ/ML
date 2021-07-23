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


dataSet, labels = creatDataSet()
tree = createTree(dataSet, labels)
print(tree)
testVec = [1, 1]
cls = myClassify(tree, labels, testVec)
print('Result = ', cls)
