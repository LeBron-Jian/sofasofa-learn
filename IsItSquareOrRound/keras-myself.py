#_*_coding:utf-8
from keras.wrappers.scikit_learn import  KerasClassifier
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    print(type(testMat.values))
    return dataMat.values , labelMat.values,testMat.values

def wider_models(X,Y):
    # 创建模型
    model = Sequential()
    # 定义一个创建神经网络模型的新函数，在这里 与13到20的基线模型相比，增加了隐藏层中神经元的数量
    model.add(Dense(20, input_dim=1600, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    # 编译模型
    model.compile(loss='mean_squared_error', optimizer='adam')
    # predict model
    model.fit(X,Y,epochs=50,batch_size=5)
    predict = model.predict(X)
    #print(pridict)
    # submit_csv = pd.read_csv(submitfile)
    # submit_csv['y'] = predict
    # submit_csv.to_csv('my_keras_prediction.csv')
    return model

if __name__  == '__main__':
    TrainFile = 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'data/sample_submit.csv'
    dataMat, labelMat, testMat = loadDataSet(TrainFile, TestFile)
    seed = 7
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp',KerasClassifier(build_fn = wider_models(dataMat,labelMat),
                                            epochs = 100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfolds = KFold(n_splits = 10, random_state = seed)
    results = cross_val_score(pipeline, dataMat, labelMat, cv= kfolds)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))