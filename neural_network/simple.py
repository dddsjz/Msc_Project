#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-18 13:18:16
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import os
from tensorflow import keras
import numpy as np

import dataprocess as dp 

dp = dataprocess.DataProcess('bidmc_01_Signals.csv')
data = dp.process_data(dp.data)
tmp = np.array(data[0])
ecg = tmp.reshape(tmp.shape[0], 1)
tmp = np.array(data[1])
ppg = tmp.reshape(tmp.shape[0], 1)


# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

print(x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=20, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)