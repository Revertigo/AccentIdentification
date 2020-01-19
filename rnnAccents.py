from __future__ import absolute_import, division, print_function, unicode_literals

import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from extractFeatures import labels
from tensorflow_core.python.keras.api._v2.keras import layers
from tensorflow_core.python.keras.layers.core import Dropout
from tensorflow_core.python.keras.losses import mean_squared_error

np.random.seed(1337)  # for reproducibility
units = 64

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_metrics(history):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        # plt.plot(history.epoch, history.history['val_' + metric],
        #          color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        # elif metric == 'accuracy':
        #     plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.show()


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
    batch_size = 30
    model = tf.keras.Sequential()

    # Reading train set
    train_data = pd.read_csv(r'resources/train_set_features.csv', header=None)

    # Retrieve features into matrix, then converting that matrix to array
    # x_train = np.array(train_data.iloc[:, 0:features].values)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(x_train)
    # # reshape x_train to fit into lstm network input
    # 'The LSTM network expects the input data (X) to be provided with a specific ' \
    # 'array structure in the form of: [samples, time steps, features].'
    # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # print("Dimesnsion 1: ", x_train.shape[1])
    # print("Dimesnsion 2: ", x_train.shape[2])
    # # Train Data labels, as array of size labels
    # y_train = np.array(train_data.iloc[:, features:].values)
    # # Reading test set
    # test_data = pd.read_csv('resources/test_set_features.csv', header=None)
    # # Retrieve features into matrix, then converting that matrix to array
    # x_test = np.array(test_data.iloc[:, 0:features].values)
    # # reshape x_train to fit into lstm input
    # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    # # Test Data labels, as array of size labels
    # y_test = np.array(test_data.iloc[:, features:].values)

    train_path = r'resources/normed_features/86_train_13_test_3_class_sliced_train'
    x_train, y_train = build_features_mat(train_path)
    print("Done reading train set...")
    test_path = r'resources/normed_features/86_train_13_test_3_class_sliced_test'
    x_test, y_test = build_features_mat(test_path)
    x_test_reshaped = x_test[:, 0, :]  # squeeze the middle dimension
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(x_test_reshaped)
    print("Done reading test set...")

    model.add(tf.keras.layers.LSTM(units, return_sequences=True, stateful=False,
                                   batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(units, return_sequences=True, stateful=False))
    model.add(tf.keras.layers.LSTM(units, stateful=False))

    # add Regularization to avoid overfitting
    model.add(Dropout(0.25))
    # Add a Dense layer with 44 units and softmax activation.
    model.add(layers.Dense(labels, activation='softmax'))

    # model = build_model(allow_cudnn_kernel=True)
    model.compile(loss='categorical_crossentropy',
                  # loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_split = 0.1, #Add validation set of 0.1
                        batch_size=batch_size,
                        verbose = 2,
                        epochs=50)

    # make predictions
    y_pred = model.predict_classes(x_test, batch_size=batch_size)
    print(y_pred)
    success = 0

    for i in range(len(y_pred)):
        real_label = 0
        for j in range(len(y_test[i])):
            if y_test[i][j] == 1:
                real_label = j
                break
        print("prediction label: ", y_pred[i])
        print("Real label: ", real_label)
        if real_label == y_pred[i]:
            success += 1

    y_real = []
    for i in range(len(y_pred)):
        real_label = 0
        for j in range(len(y_test[i])):
            if y_test[i][j] == 1:
                y_real.append(j)
                break
    print(classification_report(y_real, y_pred))

    print("\nHit count: ", success, "(out of {})".format(len(x_test)))
    print("Actual accuracy based on test set: " + str((success / len(x_test)) * 100.0) + " %")

    plot_metrics(history)
