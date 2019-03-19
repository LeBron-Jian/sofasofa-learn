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

def AlgorithmFunc(dataMat, labelMat,testMat,submitfile):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB

    trainSet, testSet, trainLabels, testLabels = train_test_split(dataMat,labelMat,
                                                               test_size=0.3,random_state=400)
    names = ['Nearest-Neighbors','Linear-SVM', 'RBF-SVM','Decision-Tree',
             'Random-Forest', 'AdaBoost', 'Naiva-Bayes', 'ExtraTrees']

    classifiers = [
        KNeighborsClassifier(),
        SVC(kernel='rbf', C=50, max_iter=5000),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        ExtraTreesClassifier(),
    ]

    for name,classifiers in zip(names, classifiers):
        classifiers.fit(trainSet, trainLabels)
        score = classifiers.score(testSet, testLabels)
        print('%s   is     %s'%(name, score))
        # from sklearn.metrics import confusion_matrix
        # test_accurcy = accuracy_score(testLabels,test_pred)
        # # test_accurcy = xgbModel.score(testSet, test_pred) * 100
        # print(" 正确率为  %s%%" %test_accurcy)
        submit = pd.read_csv(submitfile)
        submit['y'] = classifiers.predict(testMat)
        submit.to_csv('my_'+ str(name) +'_prediction.csv',index=False)

        # return test_accurcy

if __name__ == '__main__':
    TrainFile = 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'data/sample_submit.csv'
    dataMat, labelMat, testMat = loadDataSet(TrainFile,TestFile)
    AlgorithmFunc(dataMat , labelMat,testMat,SubmitFile)

