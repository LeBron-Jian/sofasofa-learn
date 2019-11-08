import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

traindata = pd.read_csv(r'data/train.csv')
testdata = pd.read_csv(r'data/test.csv')

# 去掉没有意义的一列
traindata.drop('CaseId', axis=1, inplace=True)
testdata.drop('CaseId', axis=1, inplace=True)

# 从训练集中分类标签
trainlabel = traindata['Evaluation']
traindata.drop('Evaluation', axis=1, inplace=True)

traindata1, testdata1, trainlabel1 = traindata.values, testdata.values, trainlabel.values
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(traindata1, trainlabel1,
                                                    test_size=0.3, random_state=123457)


def all_xgboost_gridsearch():
    # 分类器使用xgboost
    clf = xgb.XGBClassifier()

    # 设定网格搜索的xgboost参数搜索范围，值搜索xgboost的主要6个参数
    param_dist = {
        'n_estimators': range(80, 200, 4),
        'max_depth': range(3, 10, 1),  # 最好在3-10之间
        'learning_rate': np.linspace(0.05, 3, 20),  # 理想的学习率在0.05~3内
        'gamma': np.linspace(0.1, 0.2, 5),  # 在0.1到0.2之间就可以
        'subsample': np.linspace(0.5, 0.9, 20),  # 典型值的范围在0.5-0.9之间
        'colsample_bytree': np.linspace(0.5, 0.98, 10),
        'min_child_weight': range(1, 9, 1),
        'objective ': 'binary:logistic',
    }

    grid = GridSearchCV(clf, param_dist, cv=5, scoring='roc_auc', n_jobs=-1)

    # 在训练集上训练
    grid.fit(traindata, trainlabel)
    # 返回最优的训练器
    best_estimator = grid.best_estimator_
    print(best_estimator)


param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
param_test2 = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}

gsearch1 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=140,
        min_child_weight=4,
        # gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27),
    param_grid=param_test3,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
gsearch1.fit(X_train, y_train, )
best_estimator = gsearch1.best_estimator_
print(best_estimator)
