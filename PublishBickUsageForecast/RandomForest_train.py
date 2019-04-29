#_*_coding:utf-8_*_
import numpy as np
import pandas as pd


def load_data(trainfile, testfile):
    traindata = pd.read_csv(trainfile)
    testdata = pd.read_csv(testfile)
    feature_data = traindata.iloc[:, 1:-1]
    label_data = traindata.iloc[:, -1]
    test_feature = testdata.iloc[:, 1:]
    return feature_data, label_data, test_feature

def random_forest_train(feature_data, label_data, test_feature, submitfile):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    params = {
        'n_estimators': 70,
        'max_depth': 13,
        'min_samples_split': 10,
        'min_samples_leaf': 10,
        'max_features': 7
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)

    submit = pd.read_csv(submitfile)
    submit['y'] = model.predict(test_feature)
    submit.to_csv('my_random_forest_prediction1.csv', index=False)

def random_forest_parameter_tuning1(feature_data, label_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    param_test1 = {
        'n_estimators': range(10, 71, 10)
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        min_samples_split=100, min_samples_leaf=20, max_depth=8, max_features='sqrt',
        random_state=10), param_grid=param_test1, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_

def random_forest_parameter_tuning2(feature_data, label_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    param_test2 = {
        'max_depth': range(3, 14, 2),
        'min_samples_split': range(50, 201, 20)
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        n_estimators=70, min_samples_leaf=20, max_features='sqrt', oob_score=True,
        random_state=10), param_grid=param_test2, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_

def random_forest_parameter_tuning3(feature_data, label_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    param_test3 = {
        'min_samples_split': range(10, 90, 20),
        'min_samples_leaf': range(10, 60, 10),
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        n_estimators=70, max_depth=13, max_features='sqrt', oob_score=True,
        random_state=10), param_grid=param_test3, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_

def random_forest_parameter_tuning4(feature_data, label_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    param_test4 = {
        'max_features': range(3, 9, 2)
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        n_estimators=70, max_depth=13, min_samples_split=10, min_samples_leaf=10, oob_score=True,
        random_state=10), param_grid=param_test4, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_


if __name__ == '__main__':
    trainfile = 'data/train.csv'
    testfile = 'data/test.csv'
    submitfile = 'data/sample_submit.csv'
    feature_data, label_data, test_feature = load_data(trainfile, testfile)
    random_forest_train(feature_data, label_data, test_feature, submitfile)
