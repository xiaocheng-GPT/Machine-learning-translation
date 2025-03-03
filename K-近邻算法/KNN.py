import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()

# 预测函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]

result = classify0([0, 0], group, labels, 3)

# 实际K-近邻算法
def file2matrix(filename):
    with open(filename) as fr:
        numberOfLines = sum(1 for line in fr)
    with open(filename) as fr:
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in fr:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = list(map(float, listFromLine[0:3]))  # 将数据转换为浮点数
            classLabelVector.append(listFromLine[-1])
            index += 1
    return returnMat, classLabelVector    

# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

# label_to_numeric = {'1': 1, '2': 2, '3': 3}
# datingLabelsNum = [label_to_numeric[label] for label in datingLabels]

# fig, ax = plt.subplots()
# ax.scatter(datingDataMat[:, 0], datingDataMat[:,1], s=15.0*np.array(datingLabelsNum), c=np.array(datingLabelsNum))
# plt.show()

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# normMat, ranges, minVals = autoNorm(datingDataMat)

# 正确率计算，测试分类器
def datingClassTest():
    hoRatio = 0.2
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1
    print("the total error rate is: {}".format(errorCount / float(numTestVecs)))

# datingClassTest()

# 预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: {}".format(resultList[int(classifierResult) - 1]))

# classifyPerson()

# 图像分类
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
# result=img2vector('0_13.txt')
# print(result)

# 识别手写数字
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1
    print("the total number of errors is: {}".format(errorCount))
    print("the total error rate is: {}".format(errorCount / float(mTest)))

handwritingClassTest()