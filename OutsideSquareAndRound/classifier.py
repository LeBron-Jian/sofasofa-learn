#_*_coding:utf-8_*_
import numpy as np
from numpy import *
import pandas as pd

def load_DataSet(TrainFile,TestFile):
    traindata = pd.read_csv(TrainFile)
    testdata = pd.read_csv(TestFile)
    dataMat, labelMat = traindata.iloc[:,1:-1], traindata.iloc[:,-1]
    testMat = testdata.iloc[:,1:]
    a = traindata.isnull().sum()
    # b = testdata.isnull().sum()
    # print(a)
    # print(b)
    return dataMat, labelMat, testMat

# k-近邻算法
def classify0(inX, dataSet, labels, k):
    '''
            程式使用的是欧式距离公式
        :param inX:  用于分类的输入向量是inX
        :param dataSet:  输入的训练样本集是dataSet
        :param labels:  标签向量为labels
        :param k:  用于选择最近邻的数目
        :return:
        '''
    # shape读取数据矩阵第一维度的长度
    dataSetSize = dataSet.shape[0]
    # tile重复数组inX，有dataSet行 1个dataSet列，减法计算差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # ** 是幂运算，这里使用欧式距离
    sqDiffMat = diffMat ** 2
    # 普通sum 默认参数为axis =0 为普通相加，axis =1 为一行的行向量相加。
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort 返回数值从小到大的索引值（数组索引为0,1,2,3）
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 根据排序结果的索引值返回靠近的前k个标签
        voteLabel = labels[sortedDistIndicies[i]]
        # 各个标签出现的频率
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# Xgboost算法
def xgb_model(dataMat, labelMat, testMat, submitfile):
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    trainSet, testSet, trainLabels, testLabels = train_test_split(dataMat, labelMat,
                                                                  test_size=0.3, random_state=400)
    xgbModel = xgb.XGBClassifier(
        learning_rate= 0.1,
        n_estimators= 1000,
        max_depth= 5,
        gamma= 0,
        subsample= 0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= 4,
        seed= 32
    )
    xgbModel.fit(trainSet, trainLabels)
    # 对测试集进行预测
    test_pred = xgbModel.predict(testSet)
    print(test_pred)
    test_accuary = accuracy_score(testLabels, test_pred)
    print("正确率为 %s%%"%test_accuary)
    # submit = pd.read_csv(submitfile)
    # submit['y'] = xgbModel.predict(testMat)
    # submit.to_csv('my_XGB_prediction.csv',index=False)

# logistic 回归
def sklearn_logistic(dataMat, labelMat, testMat, submitfile):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    trainSet , testSet, trainLabels, testLabels = train_test_split(dataMat, labelMat,
                                                                   test_size = 0.3, random_state = 400)
    classifier = LogisticRegression(solver='sag', max_iter=5000)
    classifier.fit(trainSet, trainLabels)
    test_accuracy = classifier.score(testSet, testLabels) *100
    print("正确率为  %s%%"%test_accuracy)
    submit = pd.read_csv(submitfile)
    submit['y'] = classifier.predict(testMat)
    submit.to_csv('my_LR_prediction.csv',index=False)

    return test_accuracy

if __name__ == '__main__':
    TrainFile = 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'data/sample_submit.csv'
    dataMat, labelMat, testMat = load_DataSet(TrainFile,TestFile)
    # 正确率为  51.61111111111111%
    # sklearn_logistic(dataMat , labelMat,testMat,SubmitFile)
    xgb_model(dataMat, labelMat, testMat, SubmitFile)
