from __future__ import absolute_import, division, print_function, unicode_literals

import math
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from extractFeatures import labels
from tensorflow_core.python.keras.api._v2.keras import layers
from tensorflow_core.python.keras.layers.core import Dropout
from tensorflow_core.python.keras.losses import mean_squared_error

np.random.seed(1337)  # for reproducibility
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64

def build_prediction(path, ind):
    temp_y = pd.read_csv(path + "/" + str(ind) + "_pred.csv", header=None)
    return temp_y.iloc[:, 0:labels].values  # Cut the last column


def build_features_mat(path):
    train_x = []
    train_y = []
    i = 0
    for filename in os.listdir(path):
        if "_pred" not in filename:
            temp_x = pd.read_csv(path + "/" + str(i) + ".csv", header=None)
            train_x.append(temp_x.values)
            train_y.append(build_prediction(path, i))  # Cut the last column
            i += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = train_y[:, 0, :]  # Squeeze the middle dimension(from 3-dimension to 2-dimension
    return train_x, train_y


if __name__ == "__main__":
    features = 13  # Number of coefficient
    batch_size = 25
    model = tf.keras.Sequential()

    # Reading train set
    # train_data = pd.read_csv(r'resources/train_set_features.csv', header=None)
    #
    # # Retrieve features into matrix, then converting that matrix to array
    # x_train = np.array(train_data.iloc[:, 0:features].values)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(x_train)
    # # reshape x_train to fit into lstm network input
    # 'The LSTM network expects the input data (X) to be provided with a specific ' \
    # 'array structure in the form of: [samples, time steps, features].'
    # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # # Train Data labels, as array of size labels
    # y_train = np.array(train_data.iloc[:, features:].values)
    # print(x_train.shape)
    # print(y_train.shape)
    # # Reading test set
    # test_data = pd.read_csv('resources/test_set_features.csv', header=None)
    # # Retrieve features into matrix, then converting that matrix to array
    # x_test = np.array(test_data.iloc[:, 0:features].values)
    # # reshape x_train to fit into lstm input
    # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    # # Test Data labels, as array of size labels
    # y_test = np.array(test_data.iloc[:, features:].values)

    train_path = r'resources/normed_features/25_train_4_test_15_class_train'
    x_train, y_train = build_features_mat(train_path)
    print("Done reading train set...")
    test_path = r'resources/normed_features/25_train_4_test_15_class_test'
    x_test, y_test = build_features_mat(test_path)
    x_test_reshaped = x_test[:, 0, :]  # squeeze the middle dimension
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(x_test_reshaped)
    print("Done reading test set...")

    model.add(tf.keras.layers.LSTM(64, return_sequences=True, stateful=False,
                                   batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, stateful=False))
    model.add(tf.keras.layers.LSTM(64, stateful=False))

    # add Regularization to avoid overfitting
    model.add(Dropout(0.25))
    # Add a Dense layer with 44 units and softmax activation.
    model.add(layers.Dense(labels, activation='softmax'))

    # model = build_model(allow_cudnn_kernel=True)
    model.compile(loss='categorical_crossentropy',
                  # loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=batch_size,
              verbose = 2,
              epochs=50)

    # make predictions
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    success = 0

    # Check real accuracy
    for i in range(len(testPredict)):
        prediction = testPredict[i]  # returns array of predictions
        max = 0
        test_pred_label = 0
        for j in range(len(prediction)):
            if (prediction[j] > max):
                max = prediction[j]
                test_pred_label = j

        print("prediction label: ", test_pred_label)
        test_label = 0
        for j in range(len(y_test[i])):
            if y_test[i][j] == 1:
                test_label = j
                break
        print("Real label: ", test_label)
        if test_label == test_pred_label:
            success += 1


    print("\nHit count: ", success, "(out of {})".format(len(x_test)))
    print("Actual accuracy based on test set: " + str((success / len(x_test)) * 100.0) + " %")

    testScore = 0
    for i in range(y_test.shape[0]):
        testScore += math.sqrt(mean_squared_error(y_test[i], testPredict[i]))
    print('Test Score: %.3f RMSE(Root Mean Squared Error)' % (testScore / y_test.shape[0]))
    # shift test predictions for plotting
    look_back = 1
    testPredictPlot = np.empty_like(y_test)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(x_train) - 1, :] = testPredict[0]
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(testPredictPlot)
    plt.show()
