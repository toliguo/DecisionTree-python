# coding:utf-8

from math import log
import operator
from collections import Counter
import random

import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.font_manager as fm

font_path = './SimHei.ttf'
fm.fontManager.addfont(font_path)

mpl.rcParams['font.sans-serif'] = ['SimHei']
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # plt.show()


# ID3决策树
def ID3_Tree(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.title("ID3决策树", fontsize=12, color='red')
    # plt.show()
    plt.savefig('pic_results/figure_ID3.png')
    print("---------------------------------------------")
    print("png saved: pic_results/figure_ID3.png")



def read_dataset(filename):
    """
    年龄段：0代表青年，1代表中年，2代表老年；
    有工作：0代表否，1代表是；
    有自己的房子：0代表否，1代表是；
    信贷情况：0代表一般，1代表好，2代表非常好；
    类别(是否给贷款)：0代表否，1代表是
    """
    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    # print all_lines
    labels = ['年龄段', '有工作', '有自己的房子', '信贷情况']
    # featname=all_lines[0].strip().split(',')  #list形式
    # featname=featname[:-1]
    labelCounts = {}
    dataset = []
    for line in all_lines[0:]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        dataset.append(line)
    return dataset, labels


def read_testset(testfile):
    """
    年龄段：0代表青年，1代表中年，2代表老年；
    有工作：0代表否，1代表是；
    有自己的房子：0代表否，1代表是；
    信贷情况：0代表一般，1代表好，2代表非常好；
    类别(是否给贷款)：0代表否，1代表是
    """
    fr = open(testfile, 'r')
    all_lines = fr.readlines()
    testset = []
    for line in all_lines[0:]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        testset.append(line)
    return testset


# 计算信息熵
def cal_entropy(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)  # 以2为底求对数
    return Ent


# 划分数据集
def splitdataset(dataset, axis, value):
    retdataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 抽取符合划分特征的值
        if featVec[axis] == value:
            reducedfeatVec = featVec[:axis]  # 去掉axis特征
            reducedfeatVec.extend(featVec[axis + 1:])  # 将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset


'''
选择最好的数据集划分方式
ID3算法:以信息增益为准则选择划分属性
C4.5算法：使用"增益率"来选择划分属性
'''


# ID3算法
def ID3_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = cal_entropy(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        # for example in dataset:
        # featList=example[i]
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * cal_entropy(subdataset)
        infoGain = baseEnt - newEnt
        print(u"ID3中第%d个特征的信息增益为：%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # 计算最好的信息增益
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


# New function for Random Forest
def random_forest(dataset, labels, n_trees=10, sample_size=None, n_features=None):
    forest = []
    if sample_size is None:
        sample_size = len(dataset) // 2
    if n_features is None:
        n_features = int(log(len(dataset[0]) - 1, 2)) + 1

    for _ in range(n_trees):
        # Bootstrap sampling
        sampled_dataset = random.choices(dataset, k=sample_size)
        sampled_labels = labels[:]
        
        # Create tree with random feature subset
        tree = ID3_createTree(sampled_dataset, sampled_labels, n_features)
        forest.append(tree)
    
    return forest

# Modified ID3_createTree function
def ID3_createTree(dataset, labels, n_features):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    
    # Randomly select a subset of features
    feature_indices = random.sample(range(len(dataset[0]) - 1), min(n_features, len(dataset[0]) - 1))
    
    bestFeat = ID3_chooseBestFeatureToSplit(dataset, feature_indices)
    bestFeatLabel = labels[bestFeat]

    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = ID3_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            n_features)

    return ID3Tree

# Modified ID3_chooseBestFeatureToSplit function
def ID3_chooseBestFeatureToSplit(dataset, feature_indices):
    baseEnt = cal_entropy(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in feature_indices:
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEnt = 0.0
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * cal_entropy(subdataset)
        infoGain = baseEnt - newEnt
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# New function for Random Forest prediction
def random_forest_predict(forest, featLabels, testVec):
    predictions = [classify(tree, featLabels, testVec) for tree in forest]
    return max(set(predictions), key=predictions.count)

# Modified classifytest function for Random Forest
def classifytest_rf(forest, featLabels, testDataSet):
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(random_forest_predict(forest, featLabels, testVec))
    return classLabelAll

def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


def cal_acc(test_output, label):
    """
    :param test_output: the output of testset
    :param label: the answer
    :return: the acc of
    """
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    return float(count / len(test_output))


if __name__ == '__main__':
    filename = 'dataset.txt'
    testfile = 'testset.txt'
    dataset, labels = read_dataset(filename)

    print('dataset', dataset)
    print("---------------------------------------------")
    print(u"数据集长度", len(dataset))
    print("Ent(D):", cal_entropy(dataset))
    print("---------------------------------------------")

    print(u"下面开始创建随机森林-------")

    labels_tmp = labels[:]
    forest = random_forest(dataset, labels_tmp, n_trees=10)
    print(f'Random Forest created with {len(forest)} trees')
    
    testSet = read_testset(testfile)
    print("---------------------------------------------")
    print("下面为 Random Forest TestSet classifyResult：")
    rf_predictions = classifytest_rf(forest, labels, testSet)
    print(rf_predictions)
    print("---------------------------------------------")

    # Calculate accuracy
    true_labels = [example[-1] for example in testSet]
    accuracy = cal_acc(rf_predictions, true_labels)
    print(f"Random Forest Accuracy: {accuracy:.2f}")

