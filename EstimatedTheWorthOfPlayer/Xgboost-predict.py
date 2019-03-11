#_*_ coding:utf-8_*_
import pandas as pd
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

# 提取重要性特征
def RandomForestExtractFeature(traindata):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    print(type(traindata))  # <class 'numpy.ndarray'>
    X,y = traindata[:,1:-1],traindata[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    # n_estimators 森林中树的数量
    forest = RandomForestClassifier(n_estimators=100000,random_state=0,n_jobs=1)
    forest.fit(X_train,y_train.astype('int'))
    importances = forest.feature_importances_
    return importances

# 可视化展示特征重要性排序
def importances_visual(importances):
    pass


# 直接训练回归模型
def xgboost_train(traindata,testdata):
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from xgboost import plot_importance
    from matplotlib import pyplot as plt
    from sklearn.externals import joblib
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    X, y = traindata[:, 1:-1], traindata[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234567)
    # n_estimators 森林中树的数量
    model = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=160,
                             silent=True,objective='reg:gamma')
    model.fit(X_train,y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # print(y_pred)
    # 计算准确率，下面的是计算模型分类的正确率，
    MSE = mean_squared_error(y_test, y_pred)
    print("accuaracy is %s "%MSE)
    R2 = r2_score(y_test,y_pred)
    print('r2_socre is %s'%R2)
    # test_pred = model.predict(testdata[:,1:])
    # 显示重要特征
    # plot_importance(model)
    # plt.show()

    # 保存模型
    joblib.dump(model,'Model/Xgboost.m')

# 使用模型预测数据
def predict_data(xgbmodel,testdata,submitfile):
    import numpy as np
    ModelPredict = np.load(xgbmodel)
    test = testdata[:,1:]
    predict_y = ModelPredict.predict(test)
    submit_data = pd.read_csv(submitfile)
    submit_data['y'] = predict_y
    submit_data.to_csv('my_XGB_prediction.csv', index=False)
    return predict_y

if __name__ == '__main__':
    TrainFile= 'data/train.csv'
    TestFile = 'data/test.csv'
    SubmitFile = 'submit1.csv'
    xgbmodel = 'Model/Xgboost.m'
    TrainData = load_dataSet(TrainFile)
    TestData = load_dataSet(TestFile)
    # print(TrainData.shape)
    # print(TestData.shape)
    # RandomForestExtractFeature(TrainData)
    xgboost_train(TrainData,TestData)
    # print(type(Testdata))
    predict_y = predict_data(xgbmodel,TestData,SubmitFile)