# -*- coding: utf-8 -*-
import random
import re
from bayes import *

def textParse(bigString):
    reC = re.compile(r'\W')
    listOfTokens = reC.split(bigString)
    return [item.lower() for item in listOfTokens if len(item) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) # 0代表非垃圾邮件

    vocabList = createVocabList(docList)    # 去重
    trainSet = list(range(50))
    testSet = []
    for i in range(10):         # 随机选10个作为测试集
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex]) # 添加该数据的下标
        del trainSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:   # 选出训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0, p1, pSpam = trainNB0(trainMat, trainClasses)
    # 开始测试
    errorCount = 0
    for testIndex in testSet:
        vec = setOfWords2Vec(vocabList, docList[testIndex])
        if classifyNB(vec, p0, p1, pSpam) != classList[testIndex]:
            errorCount += 1
    print('the error rate is ', float(errorCount)/len(testSet))


spamTest()