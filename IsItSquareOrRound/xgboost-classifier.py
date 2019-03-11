#_*_cioding:utf-8_*_
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# load data
def loadDataSet(trainname,testname):
    '''
            对于trainSet 数据，每行前面的值分别是X1...Xn，最后一个值对应的类别标签
        :param filename:
        :return:
        '''
    datafile = pd.read_csv(trainname)
    testfile = pd.read_csv(testname)
    print(type(datafile))
    dataMat , labelMat = datafile.iloc[:,1:-1] , datafile.iloc[:,-1]
    testMat = testfile.iloc[:,1:]
    from sklearn.preprocessing import MinMaxScaler
    dataMat = MinMaxScaler().fit_transform(dataMat)
    testMat = MinMaxScaler().fit_transform(testMat)
    return dataMat , labelMat,testMat

def xgbFunc(dataMat, labelMat,testMat,submitfile):
    trainSet,testSet,trainLabels,testLabels = train_test_split(dataMat,labelMat,
                                                               test_size=0.25,random_state=400)
    xgbModel = xgb.XGBClassifier(
        # 学习率
        learning_rate=0.1,
        n_estimators=1000,
        # 构建树的深度，越大越容易过拟合
        max_depth=6,
        # 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守
        gamma=0.1,
        # 随机采样训练样本，训练实例的子采样比
        subsample= 1,
        # 生成树时进行的列采样
        colsample_bytree= 0.7,
        objective='binary:logistic',
        nthread=4,
        seed=1000,
        eta=0.2,
    )

    xgbModel.fit(trainSet, trainLabels)
    # 对测试集进行预测
    test_pred = xgbModel.predict(testSet)
    print(test_pred)
    from sklearn.metrics import confusion_matrix
    test_accurcy = accuracy_score(testLabels,test_pred)
    # test_accurcy = xgbModel.score(testLabels, test_pred) * 100
    print(" 正确率为  %.5f%%" %test_accurcy)
    submit = pd.read_csv(submitfile)
    submit['y'] = xgbModel.predict(testMat)
    submit.to_csv('my_XGB_prediction.csv',index=False)

    # return test_accurcy

if __name__ == '__main__':
    TrainFile = 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'data/sample_submit.csv'
    dataMat, labelMat, testMat = loadDataSet(TrainFile,TestFile)
    xgbFunc(dataMat , labelMat,testMat,SubmitFile)

