#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-18 13:18:16
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import os
from tensorflow import keras
import numpy as np

import dataprocess

dp = dataprocess.DataProcess('bidmc_01_Signals.csv')
data = dp.split_data(dp.data, dim = 10)
print(data[0].shape)

# Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))

# print(x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=10, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(data[0], data[0],
          epochs=10,
          batch_size=128)
score = model.evaluate(data[0], data[0], batch_size=128)