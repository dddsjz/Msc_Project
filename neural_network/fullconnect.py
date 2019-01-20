#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-23 13:53:14
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import os
import tensorflow as tf 
import numpy as np
from tensorflow import keras 
import dataprocess


class FullConnect(object):
    """

    """


    # create a full connect neural network
    def __init__(self, data, labels, val_data, val_labels):
        self.data = data
        self.labels = labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.model = keras.Sequential()

    def add_layer(self, output=7500, activation='relu', *, kernel_initializer='', 
                bias_initializaer='', kernel_regularizer='', bias_regularizer=''):
        para = 0  # ki bi kr br 0000

        if (
            (not isinstance(output, int))
            or (not isinstance(activation, str))
            or (not isinstance(kernel_initializer, (str, None)))
            or (not isinstance(bias_initializaer, (str, None)))
            or (not isinstance(kernel_regularizer, (str, None)))
            or (not isinstance(bias_regularizer, (str, None)))
        ):
            raise TypeError('Parameter Type Error')

        if kernel_initializer != '':
            para += 1
        if bias_initializaer != '':
            para += 2
        if kernel_regularizer != '' :
            para += 4
        if bias_regularizer != '' :
            para += 8

        if para == 0 :
            self.model.add(keras.layers.Dense(output, activation=activation))
        elif para == 1:
            self.model.add(
                keras.layers.Dense(output, activation=activation, 
                kernel_initializer=kernel_initializer)
            )
        elif para == 2:
            self.model.add(keras.layers.Dense(
                output, activation=activation, 
                bias_initializaer=bias_initializaer)
            )
        elif para == 4:
            self.model.add(keras.layers.Dense(
                output, activation=activation, 
                kernel_regularizer=kernel_regularizer)
            )
        elif para == 8:
            self.model.add(keras.layers.Dense(
                output, activation=activation, 
                bias_regularizer=bias_regularizer)
            )
        elif para == 5:
            self.model.add(keras.layers.Dense(
                output, activation=activation, kernel_initializer=kernel_initializer, 
                kernel_regularizer=kernel_regularizer)
            )
        elif para == 9:
            self.model.add(
                keras.layers.Dense(output, activation=activation, 
                kernel_initializer=kernel_initializer, bias_regularizer=bias_regularizer)
            )
        elif para == 6:
            self.model.add(
                keras.layers.Dense(
                output, activation=activation, bias_initializaer=bias_initializaer, 
                kernel_regularizer=kernel_regularizer)
            )
        elif para == 10:
            self.model.add(
                keras.layers.Dense(output, activation=activation, 
                bias_initializaer=bias_initializaer, bias_regularizer=bias_regularizer)
            )

    def compile_model(self, optimizer = 'tf.train.AdamOptimizer', optpara = 0.1,
        loss = 'categorical_crossentropy', metrics = 'accuracy'):
        if (
                (optimizer == 'tf.train.AdamOptimizer'
                or optimizer == 'tf.train.RMSPropOptimizer'
                or optimizer == 'tf.train.GradientDescentOptimizer')
                and isinstance(optpara, (int, float))
            ):
            tmp = optimizer + '(' + (str)(optpara) + ')'
            print(tmp)
            opt = eval(tmp)
        self.model.compile(optimizer=opt, loss=loss, metrics=[metrics])

    def run(self):
        self.model.fit(
            self.data, self.labels, epochs=10, batch_size=7500, 
            validation_data=(self.val_data, self.val_labels))


if __name__ == '__main__':

    """train set

    """
    dp = dataprocess.DataProcess('bidmc_01_Signals.csv')
    data = dp.process_data(dp.data)
    tmp = np.array(data[0])
    ecg = tmp.reshape(tmp.shape[0], 1)
    tmp = np.array(data[1])
    ppg = tmp.reshape(tmp.shape[0], 1)
    print(ecg.shape)
    print(ppg)

    """vail set

    using 2 min data to check

    """
    tmp = np.array(data[2])
    ecg_val = tmp.reshape(tmp.shape[0], 1)
    tmp = np.array(data[3])
    ppg_val = tmp.reshape(tmp.shape[0], 1)
    print(ecg_val.shape)
    print(ppg_val)

    #　test = FullConnect(ppg[0:], ecg[0:], ppg_val[0:], ecg_val[0:])
    test = FullConnect(ecg, ecg, ecg, ecg)
    test.add_layer()
    # test.add_layer()
    # test.add_layer()
    test.compile_model()
    test.run()