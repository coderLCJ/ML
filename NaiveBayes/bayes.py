from numpy import *

def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):   # 去重
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 求并集
    return list(sorted(vocabSet))

# 词集模型
def setOfWords2Vec(vocabList, inputSet):    # 返回列表 10表示词汇表中的单词是否在输入表中出现
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not my vocabulary!' % word)
    return returnVec

# 词袋模型
def bagOfWords2Vec(vocabList, inputSet):    # 返回列表 列表的值表示词汇表中的单词在输入表中出现的次数
    returnVec = [0]*len(vocabList)  # 初始化和词汇表等长的列表 存放每个词汇出现的次数
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCatgory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # 求出词汇总数 注意：每一行的元素个数都为词汇总数，因此只需求第一行的元素个数
    pAbusive = sum(trainCatgory)/float(numTrainDocs)    # 计算为侮辱性词汇的概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCatgory[i] == 1:    # 为侮辱性词汇
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   # 每个词汇出现的概率累加再加到该类别的概率的对数上
    p0 = sum(vec2Classify * p0Vec) + log(pClass1)   # #
    print(p1, p0)
    return p1 > p0

def TestingNB():
    listOPosts, listClasses = loadDataset()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postInDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postInDoc))  # 求出文档中的哪些单词在该向量出现
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, listClasses)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classify as ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))


# TestingNB()


