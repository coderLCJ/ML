import re
import feedparser
from bayes import *

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.count(word)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]  # 取出现频率最高的30个单词

def textParse(bigString):
    reC = re.compile(r'\W')
    listOfTokens = reC.split(bigString)
    return [item.lower() for item in listOfTokens if len(item) > 2]

def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minlen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)    # 去重
    top3Words = calcMostFreq(vocabList, fullText)
    for words in top3Words:                 # 去掉出现最多的三个词
        if words[0] in vocabList:
            vocabList.remove(words[0])
    print(minlen)
    trainSet = list(range(2 * minlen))
    testSet = []
    for i in range(20):         # 随机选20个作为测试集
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex]) # 添加该数据的下标
        del trainSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainSet:   # 选出训练集
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0, p1, pSpam = trainNB0(trainMat, trainClasses)    # 训练数据 得出概率
    # 开始测试
    errorCount = 0
    for testIndex in testSet:
        vec = bagOfWords2Vec(vocabList, docList[testIndex])
        if classifyNB(vec, p0, p1, pSpam) != classList[testIndex]:
            errorCount += 1
    print('the error rate is ', float(errorCount)/len(testSet))
    return vocabList, p0, p1


ny = feedparser.parse('https://newyork.craigslist.org/')
sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')
# vocabList, pSF, pNY = localWords(ny, sf)
