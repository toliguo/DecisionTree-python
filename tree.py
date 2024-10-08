# coding:utf-8

import random
from math import log
import operator

from collections import Counter, defaultdict

pre_pruning = True
post_pruning = True

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


# C4.5决策树
def C45_Tree(inTree):
    fig = plt.figure(2, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.title("C4.5决策树", fontsize=12, color='red')
    # plt.show()
    plt.savefig('pic_results/figure_C4.5.png')
    print("---------------------------------------------")
    print("png saved: pic_results/figure_C4.5.png")


# CART决策树
def CART_Tree(inTree):
    fig = plt.figure(3, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.title("CART决策树", fontsize=12, color='red')
    # plt.show()
    plt.savefig('pic_results/figure_CART.png')
    print("---------------------------------------------")
    print("png saved: pic_results/figure_CART.png")


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
C4.5算法：使用“增益率”来选择划分属性
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


# C4.5算法
def C45_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = cal_entropy(dataset)
    bestInfoGain_ratio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        IV = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * cal_entropy(subdataset)
            IV = IV - p * log(p, 2)
        infoGain = baseEnt - newEnt
        if (IV == 0):  # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV  # 这个feature的infoGain_ratio
        print(u"C4.5中第%d个特征的信息增益率为：%.3f" % (i, infoGain_ratio))
        if (infoGain_ratio > bestInfoGain_ratio):  # 选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i  # 选择最大的gain ratio对应的feature
    return bestFeature


# CART算法
def CART_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        gini = 0.0
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            subp = len(splitdataset(subdataset, -1, '0')) / float(len(subdataset))
        gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
        print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if (gini < bestGini):
            bestGini = gini
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


# 利用ID3算法创建决策树
def ID3_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = ID3_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))

    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    if pre_pruning:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)
        outputs = []
        ans = []
        for value in uniqueVals:
            cut_testset = splitdataset(test_dataset, bestFeat, value)
            cut_dataset = splitdataset(dataset, bestFeat, value)
            for vec in cut_testset:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_dataset:
                result_counter[vec[-1]] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            outputs += [leaf_output] * len(cut_testset)
        cut_acc = cal_acc(test_output=outputs, label=ans)

        if cut_acc <= root_acc:
            return leaf_output

    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = ID3_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value))

    if post_pruning:
        tree_output = classifytest(ID3Tree,
                                   featLabels=['年龄段', '有工作', '有自己的房子', '信贷情况'],
                                   testDataSet=test_dataset)
        ans = []
        for vec in test_dataset:
            ans.append(vec[-1])
        root_acc = cal_acc(tree_output, ans)
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        cut_acc = cal_acc([leaf_output] * len(test_dataset), ans)

        if cut_acc >= root_acc:
            return leaf_output

    return ID3Tree


def C45_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    C45Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    if pre_pruning:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)
        outputs = []
        ans = []
        for value in uniqueVals:
            cut_testset = splitdataset(test_dataset, bestFeat, value)
            cut_dataset = splitdataset(dataset, bestFeat, value)
            for vec in cut_testset:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_dataset:
                result_counter[vec[-1]] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            outputs += [leaf_output] * len(cut_testset)
        cut_acc = cal_acc(test_output=outputs, label=ans)

        if cut_acc <= root_acc:
            return leaf_output

    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value))

    if post_pruning:
        tree_output = classifytest(C45Tree,
                                   featLabels=['年龄段', '有工作', '有自己的房子', '信贷情况'],
                                   testDataSet=test_dataset)
        ans = []
        for vec in test_dataset:
            ans.append(vec[-1])
        root_acc = cal_acc(tree_output, ans)
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        cut_acc = cal_acc([leaf_output] * len(test_dataset), ans)

        if cut_acc >= root_acc:
            return leaf_output

    return C45Tree


def CART_createTree(dataset, labels, test_dataset):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    # print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    CARTTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    if pre_pruning:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)
        outputs = []
        ans = []
        for value in uniqueVals:
            cut_testset = splitdataset(test_dataset, bestFeat, value)
            cut_dataset = splitdataset(dataset, bestFeat, value)
            for vec in cut_testset:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_dataset:
                result_counter[vec[-1]] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            outputs += [leaf_output] * len(cut_testset)
        cut_acc = cal_acc(test_output=outputs, label=ans)

        if cut_acc <= root_acc:
            return leaf_output

    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value))

        if post_pruning:
            tree_output = classifytest(CARTTree,
                                       featLabels=['年龄段', '有工作', '有自己的房子', '信贷情况'],
                                       testDataSet=test_dataset)
            ans = []
            for vec in test_dataset:
                ans.append(vec[-1])
            root_acc = cal_acc(tree_output, ans)
            result_counter = Counter()
            for vec in dataset:
                result_counter[vec[-1]] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            cut_acc = cal_acc([leaf_output] * len(test_dataset), ans)

            if cut_acc >= root_acc:
                return leaf_output

    return CARTTree


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

import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

def CHAID_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    bestChiSquare = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        targetList = [example[-1] for example in dataset]

        # Create contingency table
        contingency_table = np.array(pd.crosstab(featList, targetList))

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        print(f"CHAID中第{i}个特征的卡方值为：{chi2:.3f}, p值为：{p_value:.3f}")

        if chi2 > bestChiSquare:
            bestChiSquare = chi2
            bestFeature = i

    return bestFeature

if __name__ == '__main__':
    filename = 'dataset.txt'
    testfile = 'testset.txt'
    dataset, labels = read_dataset(filename)

    print('dataset', dataset)
    print("---------------------------------------------")
    print(u"数据集长度", len(dataset))
    print("Ent(D):", cal_entropy(dataset))
    print("---------------------------------------------")
    print(u"以下为首次寻找最优索引:\n")
    print(u"ID3算法的最优特征索引为:" + str(ID3_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"C4.5算法的最优特征索引为:" + str(C45_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"CHAID算法的最优特征索引为:" + str(CHAID_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    from tree_rf import RF_chooseBestFeatureToSplit
    print(u"随机森林算法的最优特征索引为:" + str(RF_chooseBestFeatureToSplit(dataset)))
    print("---------------------------------------------")
    print(u"首次寻找最优索引结束！")
    print("---------------------------------------------")

    print(u"下面开始创建相应的决策树-------")

    while True:
        import sys

        args = sys.argv[1:]
        dec_tree = '1'
        if len(args) == 1:
            dec_tree = args[0]

        if dec_tree not in ['1', '2', '3', '4']:
            raise ValueError('Param must be 1, 2, 3, 4')

        # ID3决策树
        if dec_tree == '1':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            ID3desicionTree = ID3_createTree(dataset, labels_tmp, test_dataset=read_testset(testfile))
            print('ID3desicionTree:\n', ID3desicionTree)
            ID3_Tree(ID3desicionTree)
            testSet = read_testset(testfile)
            print("---------------------------------------------")
            print("下面为 ID3_TestSet_classifyResult：")
            print(classifytest(ID3desicionTree, labels, testSet))
            print("---------------------------------------------")

        # C4.5决策树
        if dec_tree == '2':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            C45desicionTree = C45_createTree(dataset, labels_tmp, test_dataset=read_testset(testfile))
            print('C45desicionTree:\n', C45desicionTree)
            C45_Tree(C45desicionTree)
            testSet = read_testset(testfile)
            print("---------------------------------------------")
            print("下面为 C4.5_TestSet_classifyResult:")
            print(classifytest(C45desicionTree, labels, testSet))
            print("---------------------------------------------")

        # CART决策树
        # if dec_tree == '3':
        #     labels_tmp = labels[:]  # 拷贝，createTree会改变labels
        #     CARTdesicionTree = CART_createTree(dataset, labels_tmp, test_dataset=read_testset(testfile))
        #     print('CARTdesicionTree:\n', CARTdesicionTree)
        #     CART_Tree(CARTdesicionTree)
        #     testSet = read_testset(testfile)
        #     print("---------------------------------------------")
        #     print("下面为 CART_TestSet_classifyResult:")
        #     print(classifytest(CARTdesicionTree, labels, testSet))
        #     print("---------------------------------------------")

        # CHAID 决策树
        if dec_tree == '3':
            from tree_CHAID import *
            labels_tmp = labels[:]
            chaid_tree = CHAID(dataset, labels_tmp)
            print('CHAIDTree:\n', chaid_tree)
            plot_CHAID_tree(chaid_tree)

            testSet = read_testset(testfile)
            print("---------------------------------------------")
            print("下面为 CHAID_TestSet_classifyResult：")
            predictions = classifytest([chaid_tree], labels, testSet)
            print(predictions)

            actual_labels = [example[-1] for example in testSet]
            accuracy = cal_acc(predictions, actual_labels)
            print(f"CHAID决策树准确率: {accuracy:.2f}")
            print("---------------------------------------------")

        # 随机森林
        if dec_tree == '4':
            from tree_rf import *

            labels_tmp = labels[:]
            random_forest_tree = random_forest(dataset, labels_tmp, n_trees=5, n_features=2)
            print('RandomForestTree:\n', random_forest_tree)
            plot_random_forest(random_forest_tree)

            testSet = read_testset(testfile)
            print("---------------------------------------------")
            print("下面为 RandomForest_TestSet_classifyResult：")
            predictions = classifytest(random_forest_tree, labels, testSet)
            print(predictions)

            # actual_labels = [example[-1] for example in testSet]
            # accuracy = cal_acc(predictions, actual_labels)
            # print(f"随机森林准确率: {accuracy:.2f}")
            print("---------------------------------------------")

        break
