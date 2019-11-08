import pandas as pd
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

traindata = pd.read_csv(r'data/train.csv')
testdata = pd.read_csv(r'data/test.csv')

# 去掉没有意义的一列
traindata.drop('CaseId', axis=1, inplace=True)
testdata.drop('CaseId', axis=1, inplace=True)

# 从训练集中分类标签
trainlabel = traindata['Evaluation']
traindata.drop('Evaluation', axis=1, inplace=True)

traindata, testdata, trainlabel = traindata.values, testdata.values, trainlabel.values

encoder = LabelEncoder()
encoder.fit(trainlabel)
encoder_label = encoder.transform(trainlabel)
label = np_utils.to_categorical(encoder_label)
# print(label)

def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=36, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=2, input_dim=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=40, batch_size=256)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(traindata, label, test_size=0.3, random_state=0)
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)
# inverse numeric variables to initial categorical labels
init_labels = encoder.inverse_transform(pred)

# k-fold cross-validate
seed = 42
np.random.seed(seed)
kflod = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, traindata, label, cv=kflod)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
