#_*_cioding:utf-8_*_
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load data
def loadDataSet(trainname,testname):
    datafile = pd.read_csv(trainname)
    testfile = pd.read_csv(testname)
    print(type(datafile))
    dataMat , labelMat = datafile.iloc[:,1:-1] , datafile.iloc[:,-1]
    testMat = testfile.iloc[:,1:]
    return dataMat , labelMat,testMat

def sklearn_logistic(dataMat, labelMat,testMat,submitfile):
    trainSet,testSet,trainLabels,testLabels = train_test_split(dataMat,labelMat,
                                                               test_size=0.2,random_state=400)
    classifier = KNeighborsClassifier()
    classifier.fit(trainSet, trainLabels)
    test_accurcy = classifier.score(testSet,testLabels) * 100
    print(" 正确率为  %.2f%%" %test_accurcy)
    submit = pd.read_csv(submitfile)
    submit['y'] = classifier.predict(testMat)
    submit.to_csv('my_KNN_prediction.csv',index=False)

    return test_accurcy

if __name__ == '__main__':
    TrainFile = 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'data/sample_submit.csv'
    dataMat, labelMat, testMat = loadDataSet(TrainFile,TestFile)
    sklearn_logistic(dataMat , labelMat,testMat,SubmitFile)

