from treePlot import *
from mainTest import *

def createData():
    lables = ['age', 'prescript', 'astigmatic', 'tearRate']
    fr = open('lenses.txt', 'r')
    retData = [data.split('\t') for data in fr.readlines()]
    return retData, lables


dataSet, labels = createData()
tree = createTree(dataSet, labels)
testvec = input('Please input you feature[\'age\', \'prescript\', \'astigmatic\', \'tearRate\']: ')
# testvec = ['young', 'yope', 'no', 'normal']
# createPlot(tree)
print('What are the contact lenses you should wear: ', classify(tree, labels, testvec.split()))
