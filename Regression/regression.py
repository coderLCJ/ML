from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
        return dataMat, labelMat

def standRegress(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr)
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:      # 计算行列式
        print('This matrix is singular, cannot reverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws



