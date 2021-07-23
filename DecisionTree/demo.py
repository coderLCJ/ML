from treePlot import *
from mainTest import *

def createData():
    lables = ['age', 'prescript', 'astigmatic', 'tearRate']
    fr = open('lenses.txt', 'r')
    data = fr.readlines()
    retData = []
    for i in range(len(data)):
        temp = data[i].split()
        if len(temp) == 6:
            temp[4] += ' ' + temp[5]
        retData.append(temp[:5])
    return retData, lables


dataSet, labels = createData()
tree = createTree(dataSet, labels)
print(tree)
createPlot(tree)