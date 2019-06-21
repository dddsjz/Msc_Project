#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-01 13:53:14
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $1.0$

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
from pywt import wavedec, waverec
from scipy.signal import butter, lfilter

class DataProcess(object):
    """Read and process PPG, ECG signal from .CSV file

    This class can Read, Split data, Split signal beat,
    Normalize signal, filte signal, Wavelet Transform,
    ICA. 

    Attributes:
        data: A pandas dataframe include PPG and ECG
        frequence: the simple frequence of signal
        minutes: the time period want to use
    """


    def __init__(self, filename, frequence = 100, minutes = 5/60, resample = 0, resample_factor = 5):
        current = os.getcwd()
        path = Path(current + '/data')
        file = path / filename
        # print(file)
        self.frequence = frequence
        self.minutes = minutes
        self.points = int(2*60*minutes*frequence)
        self.data = pd.read_csv(file, usecols=[2, 5], nrows=self.points)  # first for train, second for test
        # print(self.data)

        if resample != 0 and resample_factor != 0:
            tmp0 = self.data
            tmplit = []
            
            i = 0
            while i < self.points:
                tmplit.append(tmp0.iloc[i])
                i = i + resample_factor

            self.data = pd.DataFrame(tmplit)
            # print(pd.DataFrame(tmplit))
            self.points = int(self.points / resample_factor)

    def split_data(self, data, dim):
        """Split a pandas dataframe to individual PPG and ECG 

        Args:
            data: a pandas dataframe contains two sperate column PPG(0) & ECG(1)
            dim: divide one slice to 

        Returns:
            A list contain PPG in position 0, 2 and ECG in 1, 3. both ecg and ppg are ndarray
        """
        tmp0 = []  # return list
        tmp1 = []  # PPG train list
        tmp2 = []  # PPG test list
        tmp3 = []  # ECG train list
        tmp4 = []  # ECG test list
        split = int(self.points/2/dim)

        data = pd.DataFrame(data)

        for i in range(dim):
            tmp1.append(data.iloc[i*split:(i+1)*split, 0])  # PPG train set
            tmp2.append(data.iloc[(i+dim)*split:(i+dim + 1)*split, 0])  # PPG test set
            tmp3.append(data.iloc[i*split:(i+1)*split, 1])  # ECG train set
            tmp4.append(data.iloc[(i+dim)*split:(i+dim + 1)*split, 1])  # ECG test set

        ppg = np.asarray(tmp1)
        ppg_test = np.asarray(tmp2)
        ecg = np.asarray(tmp3)
        ecg_test = np.asarray(tmp4)

        self.ppg = ppg
        self.ecg = ecg
        self.ppg_test = ppg_test
        self.ecg_test = ecg_test
        #print(ppg.T.shape)
        #print(ecg.shape)

        tmp0.append(ppg.T) # x * dim ndarray
        tmp0.append(ecg.T)
        tmp0.append(ppg_test.T)
        tmp0.append(ecg_test.T)

        # tmp0.append(data.iloc[:split, 0])  # PPG train set
        # tmp0.append(data.iloc[:split, 1])  # ECG train set
        # tmp0.append(data.iloc[split:, 0])  # PPG test set
        # tmp0.append(data.iloc[split:, 1])  # ECG test set

        # print(data.iloc[:split, 0].shape)
        # print(data.iloc[split:, 0].shape)
        return tmp0

    def bandpass_filter(self, signal, order=5, low_feq=0.5, high_feq=40, signal_feq=125):
        """ A bandpass filter
        code referenced from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html accessed at 12/10/2018
        
        Args:
            signal: signal need to be filted
            order: the order for butterworth filter, higer order get quciker roll-off around cutoff frequency[wiki]
            low_feq: low cut off frequency
            high_feq: high cut off frequency
            signal_feq: signal sample frequency

        Returens:
            An array which is the output of the filter.
        """
        nyq_feq = 0.5 * signal_feq
        lowcut = low_feq / nyq_feq
        highcut = high_feq / nyq_feq
        b, a = butter(order, [lowcut, highcut], btype='band')
        return lfilter(b, a, signal)

    def get_median(self, data):
        n = 0
        median = data.median()
        while n < len(data):
            if data[n] == median:
                yield n
            n += 1

    def find_max(self, data):
        """Find the maximum value(e.g. R peak in ECG) for each beat

        Args:
            data: pandas dataframe contain signal(ECG or PPG)

        Returns:
            A list contain position of each max value. 
        """
        pass


    def resample(self, data):
        pass

    def standardizer(self, data):
        return scale(data)  # zero-score normalization

    def normalizer(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data)

    def wavelet_transform(self, data, level=4, wavelet_base='db4'):
        """Discrete Wavelet Transform by PyWavelets 

        Args:
            data: Origin ECG or PPG signal. 
            level: Decomposition level, If level is None then 
                it will be calculated using the dwt_max_level function.
            wavelet_base: Wavelet to use

        Returns: 
            [cA_n, cD_n, cD_n-1, …, cD2, cD1] which is an ordered list
            of coefficients arrays where n is the level of decomposition.
            Firsl element is approximation coeffients array
            and following are details coefficients arrays. 
        """
        return wavedec(data, wavelet_base, level)

    def inverse_wavelet(self, cA, cD, wavelet_base='db4', mode='smooth'):
        """
        """
        return waverec(cA, cD, wavelet_base, mode)

    def ica(self):
        pass

    def process_data_wavelet(self, data, wavelet_base='db4'):
        """Processing data with wavelet transform

        Args: origin data from xsl file

        Returns: A list include CA4, CD4, CD3, CD2 for both ppg at[0] and ecg at [1]
        """

        rst = []
        cA4s = []
        cD1s = []
        cD2s = []
        cD3s = []
        cD4s = []
        tmp = self.split_data(data, 1)
        coeff = wavedec(self.ppg.tolist(), wavelet_base, level=4)
        cA4, cD4, cD3, cD2, cD1 = coeff
        cA4s.append(np.reshape(cA4, -1))
        cD4s.append(np.reshape(cD4, -1))
        cD3s.append(np.reshape(cD3, -1))
        cD2s.append(np.reshape(cD2, -1))
        cD1s.append(np.reshape(cD1, -1))
        
        coeff = wavedec(self.ecg.tolist(), wavelet_base, level=4)
        cA4, cD4, cD3, cD2, cD1 = coeff
        cA4s.append(np.reshape(cA4, -1))
        cD4s.append(np.reshape(cD4, -1))
        cD3s.append(np.reshape(cD3, -1))
        cD2s.append(np.reshape(cD2, -1))
        cD1s.append(np.reshape(cD1, -1))

        coeff = wavedec(self.ppg_test.tolist(), wavelet_base, level=4)
        cA4, cD4, cD3, cD2, cD1 = coeff
        cA4s.append(np.reshape(cA4, -1))
        cD4s.append(np.reshape(cD4, -1))
        cD3s.append(np.reshape(cD3, -1))
        cD2s.append(np.reshape(cD2, -1))
        cD1s.append(np.reshape(cD1, -1))

        coeff = wavedec(self.ecg_test.tolist(), wavelet_base, level=4)
        cA4, cD4, cD3, cD2, cD1 = coeff
        cA4s.append(np.reshape(cA4, -1))
        cD4s.append(np.reshape(cD4, -1))
        cD3s.append(np.reshape(cD3, -1))
        cD2s.append(np.reshape(cD2, -1))
        cD1s.append(np.reshape(cD1, -1))

        rst.append(cA4s)
        rst.append(cD4s)
        rst.append(cD3s)
        rst.append(cD2s)
        rst.append(cD1s)

        return rst

    def process_data_origin(self, data, dim=10):
        """Processing data with wavelet transform

        Args: 
            data: origin data from xsl file
            dim: the slice number of dividing signal

        Returns: A ndarray for chosen slice size
        """

        data1 = self.normalizer(data)
        tmp = self.split_data(data1, dim = dim)    
        # tmp[0] = self.normalizer(tmp[0])
        # tmp[1] = self.normalizer(tmp[1])
        # tmp[2] = self.normalizer(tmp[2])
        # tmp[3] = self.normalizer(tmp[3])
        # print(tmp[0].shape())
        return tmp

class Network:
    """


    """

    def __init__(self, X, Y, nerual_number = 125, input_dim = 6):
        self.X = X 
        self.Y = Y 
        self.input_dim = input_dim
        self.nerual_number = nerual_number
        self.seed = 0

    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(units=self.nerual_number, input_dim=self.input_dim, kernel_initializer='normal', activation='relu'))
        # model.add(Dense(units=1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def create_model(self):
        # fix random seed for reproducibility
        self.seed = 7
        np.random.seed(self.seed)
        # evaluate model with standardized dataset
        estimator = KerasRegressor(build_fn=self.baseline_model(), epochs=1000, batch_size=5, verbose=0)
        return estimator

    def valiadation(self):
        kfold = KFold(n_splits=10, random_state=self.seed)
        results = cross_val_score(self.create_model(), self.X, self.Y, cv=kfold)
        print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
    
    dp = DataProcess('bidmc_01_Signals.csv')
    data = dp.data

    # map directly
    
    tmp = dp.process_data_origin(data, 5)
    
    ppg = tmp[0]
    ppg_train = tmp[2]
    ecg = tmp[1]
    ecg_train = tmp[3]
    
    # print(ppg)
    # plt.plot(ecg)
    plt.plot(ppg)
    plt.show()
    # plt.scatter(dp.ppg, ecg)
    # plt.show()

    nt = Network(ppg_train, ecg_train, 1, 5)
    m = nt.baseline_model()
    for step in range(301):
        cost = m.train_on_batch(ecg_train, ecg_train)
        if step % 100 == 0:
            print('train cost: ', cost)

    m.evaluate(ppg, ecg, batch_size=20)
    w, b = m.layers[0].get_weights()
    print('Weights=', w, '\nbiases=', b)


    Y_pred = m.predict(ppg[:200])
    # plt.scatter(ppg_train, ecg_train)
    # plt.plot(ppg_train, Y_pred)
    # plt.show()
    plt.plot(ecg)
    plt.plot(ppg)
    plt.plot(Y_pred)
    plt.show()


    # map wavelet result
    """
    tmp = dp.process_data_wavelet(data)
    
    # plt.plot(tmp[0])
    # plt.plot(tmp[1])
    # plt.show()
    plt.scatter(tmp[0][0], tmp[0][1])
    plt.show()
    
    nt = Network(tmp[0][0], tmp[0][1], 1, 1)
    
    m = nt.baseline_model()
    for step in range(1501):
       cost = m.train_on_batch(tmp[0][0], tmp[0][1])
       if step % 100 == 0:
           print('train cost: ', cost)
    m.evaluate(tmp[0][2], tmp[0][3], batch_size=2)
    w, b = m.layers[0].get_weights()
    print('Weights=', w, '\nbiases=', b)
    

    Y_pred = m.predict(tmp[0][2])
    
    plt.plot(tmp[0][3])
    plt.plot(tmp[0][2])
    plt.plot(Y_pred)
    plt.show()
    """
    """
    # resample
    dp = DataProcess('bidmc_01_Signals.csv', resample=1, resample_factor=1)
    data = dp.data

    tmp = dp.process_data_origin(data, 1)
    
    ppg = tmp[0]
    ppg_train = tmp[2]
    ecg = tmp[1]
    ecg_train = tmp[3]
    

    tmp = []
    for i in range(0, len(ppg)):
        tmp.append(i)

    time = pd.DataFrame(tmp)
    time = np.asarray(time)

    # plt.plot(ecg)
    # plt.plot(ppg)
    # plt.show()
    
    x, y, z = time, ecg, ppg
    print(x)
    print(z)
    td = plt.subplot(111, projection='3d')

    td.scatter(x, y, z)
    """