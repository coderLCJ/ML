from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt', 'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)


dataMat, labelMat = loadDataSet()
print(dataMat, '\n', labelMat)