#_*_coding:utf-8_*_

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# 导入数据，并处理非数值型数据
def load_dataSet(filename):
    '''
            检查数据，发现出生日期是00/00/00类型，
            work_rate_att  work_rate_def 是Medium High Low
            下面对这三个数据进行处理
            然后对gk
            :return:
            '''
    # 读取训练集
    traindata = pd.read_csv(filename,header=0)

    # 处理非数值型数据
    label_mapping = {
        'Low':0,
        'Medium':1,
        'High':2
    }
    traindata['work_rate_att'] = traindata['work_rate_att'].map(label_mapping)
    traindata['work_rate_def'] = traindata['work_rate_def'].map(label_mapping)

    # 将出生年月日转化为年龄
    traindata['birth_date'] = pd.to_datetime(traindata['birth_date'])
    import datetime as dt
    # 获取当前的年份
    new_year = dt.datetime.today().year
    traindata['birth_date'] = new_year - traindata.birth_date.dt.year
    print(type(traindata))

    # 处理缺失值
    res = traindata.isnull().sum()
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imr = imr.fit(traindata.values)
    imputed_data = imr.transform(traindata.values)
    return imputed_data

# define base model
def baseline_model(traindata):
    # 导入数据
    from sklearn.model_selection import train_test_split
    from sklearn.externals import joblib
    X, y = traindata[:, 1:-1], traindata[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234567)
    # create model
    model = Sequential()
    input = X.shape[1]
    print("input is %s "%input)
    model.add(Dense(128, input_shape=(input, )))
    model.add(Activation('relu'))
    model.add(Dense(1))
    # 使用高效的ADAM优化算法以及优化的最小均方误差损失函数
    model.compile(loss='mean_squared_error', optimizer=Adam())
    # early stopping
    from keras.callbacks import  EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    # train
    train_M  = model.fit(X_train,y_train,epochs=300,batch_size=20,
                         validation_data=(X_test, y_test), verbose=2,
                         shuffle=False,callbacks=[early_stopping])
    # loss 曲线
    plt.plot(train_M['loss'], label='train')
    plt.plot(train_M['val_loss'], label='test')
    plt.legend()
    # plt.show()

    # predict
    # 保存模型
    joblib.dump(model, 'Model/Keras.m')


# 使用模型预测数据
def predict_data(xgbmodel, testdata, submitfile):
    import numpy as np
    ModelPredict = np.load(xgbmodel)
    test = testdata[:, 1:]
    predict_y = ModelPredict.predict(test)
    submit_data = pd.read_csv(submitfile)
    submit_data['y'] = predict_y
    submit_data.to_csv('my_XGB_prediction.csv', index=False)
    return predict_y


def train_model(X,Y):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
    pass