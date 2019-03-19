#_*_coding:utf-8_*_
'''
    因为我使用算法的准确率最高达到99.6%，（对于这么简单的问题）
    所以打算对题目做进一步处理。
    将图片的矩阵还原看一下
    然后，对其进行二值化处理。让特征更加明显
    最后，再做算法处理。
'''

import numpy as np
import pandas as pd
from scipy.misc import imsave
from PIL import Image

def ArrayToPicture():
    # 读取数据
    train_data = pd.read_csv('data/train.csv')
    data0 = train_data.iloc[0, 1:-1]
    data1 = train_data.iloc[1, 1:-1]
    data0 = np.matrix(data0)
    data1 = np.matrix(data1)
    data0 = np.reshape(data0, (40, 40))
    print(data0)
    data1 = np.reshape(data1, (40, 40))
    imsave('test0.jpg', data0)
    imsave('test1.jpg', data1)

def PictureToArray():
    # 图像转换为矩阵
    image = Image.open('test0.jpg')
    matrix = np.asarray(image)
    print(matrix)
    # 矩阵转换为图像
    # image1 = Image.fromarray(matrix)
    # image1.show()

if __name__ =='__main__':
    ArrayToPicture()
    print('**************************************')
    PictureToArray()



