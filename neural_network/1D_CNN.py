#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-31 11:41:20
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import os
import tensorflow as tf 
import numpy as np
from tensorflow import keras 
import dataprocess
"""
class 1D_CNN(object):
    
    def __init__(self, data, labels, val_data, val_labels):
        self.data = data
        self.labels = labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.model = keras.Sequential()
"""
dp = dataprocess.DataProcess('bidmc_01_Signals.csv')
data = dp.process_data_origin(dp.data)
ecg = np.array(data[0])
#ecg = tmp.reshape(tmp.shape[0], 1)
ppg = np.array(data[1])
#ppg = tmp.reshape(tmp.shape[0], 1)
print(ecg.shape)
print(ppg)

model = keras.Sequential()
model.add(keras.layers.Conv1D(100, 250, activation='relu', input_shape=(750, 10)))
model.add(keras.layers.Conv1D(100, 125, activation='relu'))
model.add(keras.layers.MaxPooling1D(3))
model.add(keras.layers.Conv1D(160, 10, activation='relu'))
model.add(keras.layers.Conv1D(160, 10, activation='relu'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))
print(model.summary())
