import matplotlib.pyplot as plt
from trees import lensesTree

# 叶子节点代表最终决策
# 决策节点（判断节点）代表决策树中一个特征的测试
# 定义节点和箭头的样式。
# 定义plotNode函数，用于绘制节点和箭头。
# 定义plotMidText函数，用于在父子节点之间绘制文本。
# 定义plotTree函数，递归地绘制整个决策树。
# 定义createPlot函数，设置绘图环境并开始绘制决策树。
# 定义getnumLeafs和getTreeDepth函数，用于计算树的宽度和高度。
# 定义retrieveTree函数，用于获取决策树数据。
# 调用createPlot函数，传入决策树数据，开始绘制。
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
# 计算父子节点间填充的文本信息
def plotMidText(cntrpt,parentPt,txtString):
    xMid = (parentPt[0]-cntrpt[0])/2.0+cntrpt[0]
    yMid = (parentPt[1]-cntrpt[1])/2.0+cntrpt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

# 计算树的高度和宽度
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getnumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrpt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrpt, parentPt, nodeTxt)  # 标记子节点间属性值
    plotNode(firstStr, cntrpt, parentPt, decisionNode)  # 添加 nodeType 参数
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少y偏移量，避免34、35行的节点在父子节点间重叠
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrpt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrpt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrpt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

# 主函数
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getnumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    

# # 调用函数
# createPlot(lensesTree)


# 计算叶子节点和树的深度
def getnumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':          #检测节点的数据类型是否为字典，
            numLeafs += getnumLeafs(secondDict[key])          #如果是则继续递归调用，主要判断是否为叶子节点
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):       #累计计算判断节点个数，即树的深度，判断的终止条件是是否为叶子节点
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes',2:'shabi'}},3:'maybe'}}]
    return listOfTrees[i]
# myTree = retrieveTree(0)
# print(myTree)
# print(getnumLeafs(myTree))
# print(getTreeDepth(myTree))        # 测试绘画示例决策树

createPlot(lensesTree)         #绘制隐形眼镜决策树图形


