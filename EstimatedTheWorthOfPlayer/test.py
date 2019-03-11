# from sklearn import preprocessing
#
# labelEncoding = preprocessing.LabelEncoder()
# labelEncoding.fit(['Low','Medium','High'])
# res = labelEncoding.transform(['Low','Medium','High','High','Low','Low'])
# print(res)
# #    [1 2 0 0 1 1]



# import pandas as pd
#
# df = pd.DataFrame([
#             ['green', 'M', 10.1, 'class1'],
#             ['red', 'L', 13.5, 'class2'],
#             ['blue', 'XL', 15.3, 'class1']])
# print(df)
# df.columns = ['color', 'size', 'prize', 'class label']
# size_mapping = {
#     'XL':3,
#     'L':2,
#     'M':1
# }
#
# df['size'] = df['size'].map(size_mapping)
# print(df)
# class_mapping = {label:ind for ind,label in enumerate(set(df['class label']))}
# df['class label'] = df['class label'].map(class_mapping)
#
# print(df)
'''
       0   1     2       3
0  green   M  10.1  class1
1    red   L  13.5  class2
2   blue  XL  15.3  class1


   color  size  prize class label
0  green     1   10.1      class1
1    red     2   13.5      class2
2   blue     3   15.3      class1


   color  size  prize  class label
0  green     1   10.1            0
1    red     2   13.5            1
2   blue     3   15.3            0
'''
from numpy import vstack ,array ,nan
from sklearn.preprocessing import Imputer

# 缺失值计算，返回值为计算缺失值后的数据
# 参数missing

# # 创建CSV数据集
# import pandas as pd
# from io import StringIO
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 数据不要打空格，IO流会读入空格
# csv_data = '''
# A,B,C,D
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 10.0,11.0,12.0,
# '''
#
# df = pd.read_csv(StringIO(csv_data))
# print(df)
#
# # 均值填充
# from sklearn.preprocessing import Imputer
# # axis = 0 表示列向 ，采用每一列的平均值填充空值
# imr = Imputer(missing_values='NaN',strategy='0',axis=0)
# imr = imr.fit(df.values)
# imputed_data = imr.transform(df.values)
# print(imputed_data)
'''
      A     B     C    D
0   1.0   2.0   3.0  4.0
1   5.0   6.0   NaN  8.0
2  10.0  11.0  12.0  NaN

[[ 1.   2.   3.   4. ]
 [ 5.   6.   7.5  8. ]
 [10.  11.  12.   6. ]]
 '''

# a = '18/01/00'
# def full_birthDate(x):
#     if(x[-2:]) <= '50':
#         return x[:-2] + '20' + x[-2:]
#     else:
#         return x[:-2] + '19' + x[-2:]
#
# a = full_birthDate(a)
# print(a)、、

import datetime as dt

# # 获取当前的年份
# new_year = dt.datetime.today()
# print(new_year)


# 体重指数（BMI） = 体重（kg） / 身高^2 (m)
# 当BMI指数为18.5～23.9时属正常。
weight = 75
height = 1.76
BMI = weight / (height **2)
print(BMI)

trai = 10000. * weight / (height ** 2)

print(trai)
import math
a = math.nan
if a>0:
    print("ok")
else:
    print("NG")
