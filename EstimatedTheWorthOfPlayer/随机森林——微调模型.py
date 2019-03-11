#_*_coding:utf-8_*_
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor

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

# 获取球员最擅长位置的评分
positions = ['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb','gk']

train['best_pos'] = train[positions].max(axis=1)
test['best_pos'] = test[positions].max(axis=1)

# 计算球员的身体质量指数（BMI）
train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)

# 判断一个球员是否是守门员
train['is_gk'] = train['gk'] > 0
test['is_gk'] = test['gk'] > 0

# 用多个变量准备训练随机森林
test['pred'] = 0
cols =  ['height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation', 'age', 'best_pos']

# 用非守门员的数据训练随机森林
# 使用网格搜索微调模型
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10],'max_features':[2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid ,cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(train[train['is_gk'] == False][cols] , train[train['is_gk'] == False]['y'])

preds = grid_search.predict(test[test['is_gk'] == False][cols])
test.loc[test['is_gk'] == False , 'pred'] = preds

# 用守门员的数据训练随机森林
# 使用网格搜索微调模型
from sklearn.model_selection import GridSearchCV
'''
先对estimators进行网格搜索，[3,10,30]
接着对最大深度max_depth 
内部节点再划分所需要最小样本数min_samples_split 进行网格搜索
最后对最大特征数,ax_features进行调参
'''
param_grid1 = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10],'max_features':[2,3,4]}
]
'''
parm_grid 告诉Scikit-learn 首先评估所有的列在第一个dict中的n_estimators 和
max_features的 3*4=12 种组合，然后尝试第二个dict中的超参数2*3 = 6 种组合，
这次会将超参数bootstrap 设为False 而不是True（后者是该超参数的默认值）

总之，网格搜索会探索 12 + 6 = 18 种RandomForestRegressor的超参数组合，会训练
每个模型五次，因为使用的是五折交叉验证，换句话说，训练总共有18 *5 = 90 轮，、
将花费大量的时间，完成后，就可以得到参数的最佳组合了
'''
forest_reg1 = RandomForestRegressor()
grid_search1 = GridSearchCV(forest_reg, param_grid1 ,cv=5,
                           scoring='neg_mean_squared_error')

grid_search1.fit(train[train['is_gk'] == True][cols] , train[train['is_gk'] == True]['y'])

preds = grid_search1.predict(test[test['is_gk'] == True][cols])
test.loc[test['is_gk'] == True , 'pred'] = preds

# 输出预测值
submit['y'] = np.array(test['pred'])
submit.to_csv('my_RF_prediction1.csv',index = False)

# 打印参数的最佳组合
print(grid_search.best_params_)