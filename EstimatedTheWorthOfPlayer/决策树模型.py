#_*_coding:utf-8_*_
import  pandas as pd
import datetime as dt
from sklearn.tree import DecisionTreeRegressor

# 读取数据
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')


# 获得球员年龄
today = dt.datetime.today().year

train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = today - train.birth_date.dt.year

test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = today - test.birth_date.dt.year

# 获得球员最擅长位置上的评分
positions = ['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb','gk']

train['best_pos'] = train[positions].max(axis =1)
test['best_pos'] = test[positions].max(axis = 1)

# 用潜力，国际知名度，年龄，最擅长位置评分 这四个变量来建立决策树模型
col = ['potential','international_reputation','age','best_pos']

reg = DecisionTreeRegressor(random_state=100)
reg.fit(train[col],train['y'])

# 输出预测值
submit['y'] = reg.predict(test[col])
submit.to_csv('my_DT_prediction.csv',index=False)