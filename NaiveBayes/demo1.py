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
    trainSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del trainSet[randIndex]


trainSet = list(range(50))
print(trainSet)
del trainSet[1]
print(trainSet)