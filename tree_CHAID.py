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


def plotNode(nodeTxt, centerPt, parentPt, nodeType, ax):
    ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                xytext=centerPt, textcoords='axes fraction',
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


def plotMidText(cntrPt, parentPt, txtString, ax):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    ax.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt, ax):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt, ax)
    plotNode(firstStr, cntrPt, parentPt, decisionNode, ax)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key), ax)
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode, ax)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key), ax)
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


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


def chi_square_test(dataset, feature_index):
    # 计算卡方值
    observed = Counter((example[feature_index], example[-1]) for example in dataset)
    classes = set(example[-1] for example in dataset)
    feature_values = set(example[feature_index] for example in dataset)
    
    expected = {}
    for value in feature_values:
        for cls in classes:
            row_total = sum(observed[(value, c)] for c in classes)
            col_total = sum(observed[(v, cls)] for v in feature_values)
            total = len(dataset)
            expected[(value, cls)] = (row_total * col_total) / total

    chi_square = sum((observed[(value, cls)] - expected[(value, cls)]) ** 2 / expected[(value, cls)]
                     for value in feature_values for cls in classes if expected[(value, cls)] != 0)
    
    return chi_square

def CHAID_tree(dataset, labels, alpha=0.05):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)

    best_feature = -1
    best_chi_square = 0
    for i in range(len(dataset[0]) - 1):
        chi_square = chi_square_test(dataset, i)
        if chi_square > best_chi_square:
            best_chi_square = chi_square
            best_feature = i

    # 使用scipy库计算p值
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(best_chi_square, (len(set(example[best_feature] for example in dataset)) - 1) * (len(set(classList)) - 1))

    if p_value > alpha:
        return majorityCnt(classList)

    best_feat_label = labels[best_feature]
    tree = {best_feat_label: {}}
    del(labels[best_feature])

    feat_values = [example[best_feature] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        tree[best_feat_label][value] = CHAID_tree(splitdataset(dataset, best_feature, value), sub_labels, alpha)

    return tree

def CHAID(dataset, labels, alpha=0.05):
    return CHAID_tree(dataset, labels[:], alpha)

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


def classifytest(forest, featLabels, testDataSet):
    predictions = []
    for testVec in testDataSet:
        votes = [classify(tree, featLabels, testVec) for tree in forest]
        prediction = max(set(votes), key=votes.count)
        predictions.append(prediction)
    return predictions


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

def plot_CHAID_tree(tree):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("CHAID决策树", fontsize=16, color='red')

    plotTree.totalw = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '', ax)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('pic_results/figure_CHAID.png')
    print("---------------------------------------------")
    print("png saved: pic_results/figure_CHAID.png")


if __name__ == '__main__':
    filename = 'dataset.txt'
    testfile = 'testset.txt'
    dataset, labels = read_dataset(filename)

    print('dataset', dataset)
    print("---------------------------------------------")
    print(u"数据集长度", len(dataset))
    print("Ent(D):", cal_entropy(dataset))
    print("---------------------------------------------")

    print(u"下面开始创建CHAID决策树-------")

    labels_tmp = labels[:]
    chaid_tree = CHAID(dataset, labels_tmp)
    print('CHAID Tree created')
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
