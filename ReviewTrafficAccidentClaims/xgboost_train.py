import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

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
# 训练模型
model = xgb.XGBClassifier(max_depth=6,
                          learning_rate=0.1,
                          min_child_weight=4,
                          gamma=0.3,
                          n_estimators=5000,
                          silent=True,
                          objective='binary:logistic',
                          nthread=4,
                          seed=27,
                          scale_pos_weight=1,
                          subsample=0.9,
                          colsample_bytree=0.6,
                          reg_alpha=1,
                          )

model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:%2.f%%' % (accuracy * 100))


# 查看AUC评价标准
# from sklearn import metrics
##必须二分类才能计算
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_pred))

def run_predict():
    y_pred_test = model.predict_proba(testdata1)[:, 1]
    # 保存预测的结果
    submitData = pd.read_csv(r'data/sample_submit.csv')
    submitData['Evaluation'] = y_pred_test
    submitData.to_csv("xgboost.csv", index=False)

run_predict()
