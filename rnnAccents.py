
from __future__ import absolute_import, division, print_function, unicode_literals

# import matplotlib
import os

import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow_core.python.keras.api._v2.keras import layers
from tensorflow_core.python.keras.layers.core import Dropout

batch_size = 64
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64
labels = 44

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))
  model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(labels, activation='softmax')]
  )
  return model

if __name__ == "__main__":

    features = 13  # Number of coefficient
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 13, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=13, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))
    # Add regularization to out model
    #model.add(Dropout(0.5))
    # Add a Dense layer with 44 units and softmax activation.
    model.add(layers.Dense(labels, activation='softmax'))

    train_data = pd.read_csv(r'resources/train_set_features.csv', header=None)

    # Retrieve features into matrix, then converting that matrix to array
    x_train = np.array(train_data.iloc[:, 0:features].values)  # Batch Gradient Descent - using the whole data set
    #reshape x_train to fit into lstm input
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # Train Data labels, as array of size labels
    y_train = np.array(train_data.iloc[:, features:].values)

    print(y_train)

    # Reading test set
    test_data = pd.read_csv('resources/test_set_features.csv', header=None)

    # Retrieve features into matrix, then converting that matrix to array
    x_test = np.array(test_data.iloc[:, 0:features].values)
    # reshape x_train to fit into lstm input
    #x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # Test Data labels, as array of size labels
    y_test = np.array(test_data.iloc[:, features:].values)

    #print(y_test)

    #model = build_model(allow_cudnn_kernel=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=batch_size,
              epochs=5)

    y_pred = model.predict_classes(x_test, batch_size=batch_size)
    print(y_pred)
