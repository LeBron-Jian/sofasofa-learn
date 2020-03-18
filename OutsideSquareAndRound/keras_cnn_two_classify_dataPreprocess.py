# _*_coding:utf-8_*_
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Convolution2D
from keras.models import Sequential
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, misc
import statistics


def load_train_test_data(train, test):
    np.random.shuffle(train)
    # train = train[:, :-1]
    labels = train[:, -1]
    test = np.array(test)
    # print(train.shape, test.shape, labels.shape)  # (6000, 1600) (5191, 1600) (6000,)

    data, data_test = data_modify_suitable_train(train, True), data_modify_suitable_train(test, False)
    # print(data.shape, data_test.shape)

    # train = train.reshape([train.shape[0], 40, 40])
    # test = test.reshape([test.shape[0], 40, 40])
    # data, data_test = batch_preprocessing(train), batch_preprocessing(test)
    # # print(data.shape, data_test.shape)  # (6000, 40, 40) (5191, 40, 40)

    # 注意这里需要添加一个维度
    data, data_test = np.expand_dims(data, 3), np.expand_dims(data_test, 3)
    # print(data.shape, data_test.shape)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.33)  # test_size=0.7
    return train_x, train_y, test_x, test_y, data_test


def data_test(train, test):
    np.random.shuffle(train)
    train = train[:, :-1]
    labels = train[:, -1]
    test = np.array(test)
    # print(train.shape, test.shape, labels.shape)  # (6000, 1600) (5191, 1600) (6000,)
    train = train.reshape([train.shape[0], 40, 40])
    test = test.reshape([test.shape[0], 40, 40])
    data = my_preprocessing(train[1])
    all_data = batch_preprocessing(data_set=train)
    # data = data_modify_suitable_train(train, True)
    print(data.shape, type(data))  # (40, 40) <class 'numpy.ndarray'>
    print(all_data.shape, type(all_data))
    plt.gray()
    plt.imshow(all_data[1])
    plt.axis('off')
    plt.title('origin photo')


def my_preprocessing(I, show_fig=False):
    I_median = ndimage.median_filter(I, size=5)
    mask = (I_median < statistics.mode(I_median.flatten()))
    I_out = scipy.ndimage.morphology.binary_closing(mask, iterations=2)
    if (np.mean(I_out[15:25, 15:25].flatten()) < 0.5):
        I_out = 1 - I_out

    if show_fig:
        fig = plt.figure(figsize=(8, 4))
        plt.gray()
        plt.subplot(2, 4, 1)
        plt.imshow(I)
        plt.axis('off')
        plt.title('Image')
        plt.subplot(2, 4, 2)

        plt.imshow(I_median)
        plt.axis('off')
        plt.title('Median filter')

        plt.subplot(2, 4, 3)
        plt.imshow(mask)
        plt.axis('off')
        plt.title('Mask')

        plt.subplot(2, 4, 4)
        plt.imshow(I_out)
        plt.axis('off')
        plt.title('Closed mask')
        fig.tight_layout()
        plt.show()
    return I_out


def batch_preprocessing(data_set):
    zero_data = np.zeros_like(data_set)
    data_n = data_set.shape[0]
    for i in range(data_n):
        zero_data[i] = my_preprocessing(data_set[i])
    return zero_data


def data_modify_suitable_train(data_set=None, type=True):
    if data_set is not None:
        data = []
        if type is True:
            np.random.shuffle(data_set)
            # data = data_set[:, 0: data_set.shape[1] - 1]
            data = data_set[:, 0: -1]
            print(data.shape)
        else:
            data = data_set
    data = np.array([np.reshape(i, (40, 40)) for i in data])
    print('data', data.shape)
    # median_data = np.array([median_filter(i, size=(5, 5)) for i in data])
    # print('median', median_data.shape)
    # mask = (median_data < statistics.mode(median_data.flatten()))
    # print('mask', mask.shape)
    # res_data = ndimage.morphology.binary_closing(mask, iterations=2)
    # if (np.mean(res_data[15:25, 15:25].flatten()) < 0.5):
    #     res_data = 1 - res_data
    # print('res_data', res_data.shape)
    zero_data = np.zeros_like(data)
    data_n = data.shape[0]
    for i in range(data_n):
        zero_data[i] = my_preprocessing(data[i])
    return zero_data

    # data = np.array([(i > 10) * 100 for i in data])
    # data = np.array([np.reshape(i, (i.shape[0], i.shape[1], 1)) for i in res_data])
    # data = np.array([np.reshape(i, (i.shape[0], i.shape[1])) for i in res_data])
    # return data


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def built_model():
    model = Sequential()
    model.add(Convolution2D(filters=8,
                            kernel_size=(5, 5),
                            input_shape=(40, 40, 1),
                            activation='relu'))
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1,
                    activation='sigmoid'))
    # 完成模型的搭建后，使用.compile方法来编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])
    model.summary()
    return model


def built_model_plus():
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
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    model.summary()
    return model


def train_models(train, test, batch_size=64, epochs=20, model=None):
    train_x, train_y, test_x, test_y, t = load_train_test_data(train, test)
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape, t.shape)
    if model is None:
        # model = built_model()
        model = built_model_plus()
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2)
        print("刻画损失函数在训练与验证集的变化")
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()
        #   ##### 注意这个t
        pred_prob = model.predict(t, batch_size=batch_size, verbose=1)
        # pred = np.array(pred_prob > 0.5).astype(int)
        # score = model.evaluate(test_x, test_y, batch_size=batch_size)
        # print('score is %s' % score)
        # print("刻画预测结果与测试结果")
        for i in range(pred_prob.shape[0]):
            if pred_prob[i][0] > 0.7:
                pred_prob[i][0] = 1
            elif pred_prob[i][0] < 0.3:
                pred_prob[i][0] = 0
            else:
                pred_prob[i][0] = 2
        return pred_prob.astype(int)


if __name__ == '__main__':
    trainFile = 'dataout/train.csv'
    testFile = 'dataout/test.csv'
    submitfile = 'dataout/sample_submit.csv'
    train = pd.read_csv(trainFile)
    test = pd.read_csv(testFile)
    train = np.array(train.drop('id', axis=1))
    test = np.array(test.drop('id', axis=1))
    # print(train.shape, test.shape)  # (6000, 1601) (5191, 1600)

    # load_train_test_data(train, test)

    # data test
    # data_test(train, test)
    # load_train_test_data(train, test)
    # print('over')

    pred = train_models(train, test)
    submit = pd.read_csv('dataout/sample_submit.csv')
    submit['y'] = pred
    submit.to_csv('my_CNN_prediction1.csv', index=False)
