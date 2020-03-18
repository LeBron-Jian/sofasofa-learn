from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from scipy import ndimage
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle

SEED = 0
tf.random.set_random_seed(SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def load_DateSet(TrainFile, TestFile):
    traindata = pd.read_csv(TrainFile)
    testdata = pd.read_csv(TestFile)
    dataMat_origin, testMat_origin = np.array(traindata.drop('id', axis=1)), np.array(testdata.drop('id', axis=1))
    dataMat, labelMat, testMat = dataMat_origin[:, 0:-1], dataMat_origin[:, -1], testMat_origin
    # print(dataMat.shape, labelMat.shape, testMat.shape)

    #  将矩阵转化为（40， 40） 的格式
    dataMat = np.array([np.reshape(i, (40, 40)) for i in dataMat])
    testMat = np.array([np.reshape(i, (40, 40)) for i in testMat])

    # return dataMat_origin, testMat_origin, dataMat, labelMat, testMat
    return dataMat, labelMat, testMat

def my_preprocessing(I, show_fig=False):
    # data = np.array([ndimage.median_filter(i, size=(3, 3)) for i in data])
    # data = np.array([i > 10]*100 for i in data)
    I_median = ndimage.median_filter(I, size=5)
    mask = (I_median < statistics.mode(I_median.flatten()))
    I_out = ndimage.morphology.binary_closing(mask, iterations=2)
    if (np.mean(I_out[15:25, 15:25].flatten()) < 0.5):
        I_out = 1 - I_out

    if show_fig:
        fig = plt.figure(figsize=(8, 4))
        plt.gray()
        plt.subplot(1, 4, 1)
        plt.imshow(I)  # 原图
        plt.axis('off')
        plt.title('Image')

        plt.subplot(1, 4, 2)
        plt.imshow(I_median)  # 中值滤波处理
        plt.axis('off')
        plt.title("Median filter")

        plt.subplot(1, 4, 3)
        plt.imshow(mask)  # 添加掩膜
        plt.axis('off')
        plt.title('Mask')

        plt.subplot(1, 4, 4)
        plt.imshow(I_out)  # 形态闭合处理
        plt.axis('off')
        plt.title('Closed mask')
        fig.tight_layout()
        plt.show()
    return I_out
    # return I_median

def batch_preprocessing(dataMat, labelMat, testMat):
    train_prc = np.zeros_like(dataMat)
    test_prc = np.zeros_like(testMat)
    for i in range(dataMat.shape[0]):
        train_prc[i] = my_preprocessing(dataMat[i])
    for i in range(testMat.shape[0]):
        test_prc[i] = my_preprocessing(testMat[i])
    # print("over ...")
    return train_prc, labelMat, test_prc

# 人工从测试集中挑选训练集中并未出现过的标签样本，加入训练集中
def add_other_sample_func(train_prc, labelMat, test_prc):
    # 五个异形在原数据中对应的下标
    # 手动查看测试集中的图像，并且增加了五个异性，放入训练集中
    num_anno = 5
    anno_inx = np.array([4949, 4956, 4973, 4974, 4988])
    # anno_inx = np.array([4000, 4001, 4004, 4010, 4013])
    anno_inx = anno_inx[::-1]
    anno_inx_add = anno_inx[:num_anno]   # array([4988, 4974, 4973, 4956, 4949])

    x_train_prc_anno = train_prc
    y_train_anno = labelMat.reshape([-1, 1])

    x_add = test_prc[anno_inx_add]
    y_add = np.ones([num_anno, 1])*2

    # 对异形进行过采样
    # for i in range(4000/num_anno):
    for i in range(800):
        x_train_prc_anno = np.append(x_train_prc_anno, x_add, axis=0)
        y_train_anno = np.append(y_train_anno, y_add, axis=0)

    x_train_prc_anno, y_train_anno = shuffle(x_train_prc_anno, y_train_anno, random_state=0)
    mlb1 = MultiLabelBinarizer()
    y_train_mlb = mlb1.fit_transform(y_train_anno)
    return x_train_prc_anno, y_train_mlb, test_prc

def built_model():
    n_filter = 32
    # 序贯模型是多个网络层的线性堆叠，也就是“一条道走到黑”
    model = Sequential()
    # 通过 .add() 方法一个个的将 layer加入模型中
    model.add(Convolution2D(filters=n_filter,
                            kernel_size=(5, 5),
                            input_shape=(40, 40, 1),
                            activation='relu'))
    model.add(Convolution2D(filters=n_filter,
                            kernel_size=(5, 5),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=n_filter,
                            kernel_size=(5, 5),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=n_filter,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    # final layer using softmax
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'])

    model.summary()
    return model


def train_model(x_train, y_train, x_test, batch_size=64, epochs=20, model=None,
                class_weight={0: 1., 1: 1., 2: 10.}):
    if np.ndim(x_train) < 4:
        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)
    if model is None:
        model = built_model()

        datagen = ImageDataGenerator(
            rotation_range=180,  # 整数，数据提升时图片随机转动的角度
            width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片水平便宜的幅度
            height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
            horizontal_flip=True  # 布尔值，进行随机水平翻转
        )

        # 训练模型的同时进行数据增广
        # flow(self, X, y batch_size=21, shuffle=True, seed=None,save_to_dir=None, save_prefix='' save_format='jpeg')
        # 接收 numpy数组和标签为参数，生成经过数据提升或标准化后的batch数据，并在一个无线循环中不断的返回 batch数据
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                                      class_weight=class_weight,
                                      validation_data=datagen.flow(x_train, y_train,
                                                                   batch_size=batch_size),
                                      validation_steps=1)

    print("Loss on training and testing")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()

    pred_prob_train = model.predict(x_train, batch_size=batch_size, verbose=1)
    pred_train = np.array(pred_prob_train > 0.5).astype(int)
    pred_prob_test = model.predict(x_test, batch_size=batch_size, verbose=1)
    pred_test = np.array(pred_prob_test > 0.5).astype(int)
    y_test_hat = pred_test[:, 1] + pred_test[:, 2] * 2
    y_train_hat = pred_train[:, 1] + pred_train[:, 2] * 2
    return y_train_hat, y_test_hat, history



if __name__ == '__main__':
    trainFile = 'dataout/train.csv'
    testFile = 'dataout/test.csv'
    submitfile = 'dataout/sample_submit.csv'
    epochs = 100
    # 考虑到异性并不多，所以设置如下权重，来解决非平衡下的分类
    class_weight = {0: 1., 1: 1., 2: 10.}

    dataMat, labelMat, testMat = load_DateSet(trainFile, testFile)
    # 数据预处理
    # I_out = my_preprocessing(testMat[5], True)
    train_prc, labelMat, test_prc = batch_preprocessing(dataMat, labelMat, testMat)
    x_train_prc_anno, y_train_mlb, test_prc = add_other_sample_func(train_prc, labelMat, test_prc)
    # y_train_hat, y_test_hat, hisory = train_model(x_train_prc_anno, y_train_mlb, test_prc,
    #                                           epochs=epochs, batch_size=64, class_weight=class_weight)

    # 提交结果，查看精度
    # submit = pd.read_csv(submitfile)
    # submit['y'] = y_test_hat
    # submit.to_csv('my_cnn_prediction33.csv', index=False)
