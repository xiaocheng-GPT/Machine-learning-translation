from numpy import *
import re
import feedparser
import operator



'''
代码作用为：创建一个集合，将集合添加到新的空集合中，并去除出重复项，最后返回打印一个列表
注：集合的去重，使用集合的|运算符，集合是无序的，且不能通过索引访问，只能通过迭代访问

'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
# # 测试
# dataSet, classVec = loadDataSet()
# vocabList = createVocabList(dataSet)
# print(vocabList)
# print(setOfWords2Vec(vocabList, dataSet[3]))


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 0.0; 
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# # 测试
# dataSet, classVec = loadDataSet()
# vocabList = createVocabList(dataSet)
# trainMat = []
# for doc in dataSet:
#     trainMat.append(setOfWords2Vec(vocabList, doc))
# p0V, p1V, pSpam = trainNB0(trainMat, classVec)
# print(p0V)
# print(p1V)
# print(pSpam)
# print(vocabList)


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAbusive = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAbusive))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAbusive))
# testingNB()


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# # 使用朴素贝叶斯模型过滤垃圾
# mySend='this book is the best book on python or M.L. I have ever laid eyes upon.'
# # print(mySend.split())
# result = re.split(r'\W+', mySend)    #正则表达式切分字符串
# # print(result)

# # 过滤掉长度为0的空字符串，并转换为小写
# filtered_result = [token.lower() for token in result if len(token) > 0]

# print(filtered_result)
# email/ham/6.txt测试
# emailText = open('email/spam/6.txt').read()
# listOfTokens=re.split(r'\W+',emailText)
# # print(listOfTokens)

# result=[token.lower() for token in listOfTokens if len(token)>3]
# print(result)



# 朴素贝叶斯：文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
def spamTest(file_path):
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        with open(file_path + '/ham/%d.txt' % i, encoding='ISO-8859-1') as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        with open(file_path + '/spam/%d.txt' % i, encoding='ISO-8859-1') as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pAb = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pAb) != classList[docIndex]:
            errorCount += 1
    print('classification error rate:', float(errorCount) / len(testSet))

# spamTest('email')

# ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# ny['entries']
# print(len(ny['entries']))

# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):     #该函数用于计算并返回 vocabList 中每个词汇在 fullText 中的出现频率，最终返回出现频率最高的前30个词。

    freqDict = {}
    for token in vocabList:                             #vocabList: 一个词汇列表。
        freqDict[token] = fullText.count(token)         #fullText: 一个包含所有文本的列表。
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

'''
创建一个空字典 freqDict 用来存储词汇和它们的频率。
遍历 vocabList 中的每个词汇，计算它在 fullText 中出现的次数，并将其存储在字典中。
使用 sorted() 函数按照频率排序，返回按频率从高到低排序后的前30个词汇。
'''

def localWords(feed1, feed0):
    import feedparser
    #初始化变量
    docList=[]; classList = []; fullText =[]
    #数据预处理
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)            #docList：用来存储所有处理后的词汇列表。
        fullText.extend(wordList)          #fullText：包含所有文档中的所有词汇，用于高频词分析。
        classList.append(1)                 #classList：用来存储每个文档的类别标签（1 或 0）。1 表示垃圾邮件，0 表示非垃圾邮件。
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #创建词汇表
    vocabList = createVocabList(docList)
    #去除高频词汇
    top30Words = calcMostFreq(vocabList, fullText)
    for pairWise in top30Words:
        if pairWise[0] in vocabList:
            vocabList.remove(pairWise[0])
    #划分训练集和测试集
    trainingSet = list(range(2*minLen)); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #构造训练数据
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #训练朴素贝叶斯分类器
    p0v, p1v, pAb = trainNB0(array(trainMat), array(trainClasses))
    #测试模型并计算错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector =(bagOfWords2VecMN(vocabList, docList[docIndex]))
        if classifyNB(array(wordVector), p0v, p1v, pAb) != classList[docIndex]:
            errorCount += 1
    print('classification error rate:', float(errorCount) / len(testSet))
    #返回结果
    return vocabList,p0v,p1v


ny=feedparser.parse(' http://news.baidu.com/n?cmd=1&class=nba&tn=rss&sub=0/index.rss')
sf=feedparser.parse('http://news.baidu.com/n?cmd=1&class=cba&tn=rss&sub=0/index.rss')

# vocabList,pSF,pNY=localWords(ny,sf)
# print(vocabList)



# 最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0v,p1v=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            topSF.append((vocabList[i],p0v[i]))
        if p1v[i] > -6.0:
            topNY.append((vocabList[i],p1v[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF:SF**SF**SF")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY:NY**NY**NY")
    for item in sortedNY:
        print(item[0])
getTopWords(ny,sf)
