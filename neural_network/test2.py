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
import random

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
        frequency: the simple frequency of signal
        minutes: the time period want to use
    """


    def __init__(self, filename, frequency = 100, minutes = 100/60, resample = 0, resample_factor = 5):
        current = os.getcwd()
        path = Path(current + '/data')
        file = path / filename
        # print(file)
        self.frequency = frequency
        self.minutes = minutes
        self.points = int(2*60*minutes*frequency)
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

    def calc(self, data1, data2):
        """
        Work for resample function.
        """
        if data1 >= data2:
            return (data1 - data2) / 2 + data2
        else:
            return (data2 - data1) / 2 + data1

    def resample(self, signal, aim_feq):
        # Todo: delete point to the aim_feq
        # Debug for the add pint method, find the reason for unwanted point value
        
        tmp = signal.tolist()
        # print(tmp[40])
        # print(tmp[41])


        while(len(tmp) < aim_feq-1):
        # i = 0
        # while(i < 1):
             # padding
            rad = random.randint(1, int(len(tmp)/3)-1)
            f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)
            # print(f)
            # print(tmp[f])
            # print(tmp[f-1])
            mean_f = self.calc(tmp[f], tmp[f-1])
            mean_m = self.calc(tmp[m], tmp[m-1])
            mean_l = self.calc(tmp[l], tmp[l-1])
            # print(mean_f)

            tmp.insert(f, mean_f)
            # print(tmp[f])
            # print(tmp[f+1])
            # print(tmp[f-1])
            tmp.insert(m, mean_m)
            tmp.insert(l, mean_l)

            # i = i + 1

        rad = random.randint(1, int(len(tmp)/3)-1)
        f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)

        if abs(len(tmp)-aim_feq) % 3 == 1:

            mean_m = self.calc(tmp[m], tmp[m-1])

            tmp.insert(m, mean_m)

        elif abs(len(tmp)-aim_feq) % 3 == 2:

            mean_f = self.calc(tmp[f], tmp[f-1])
            mean_l = self.calc(tmp[l], tmp[l-1])

            tmp.insert(f, mean_f)
            tmp.insert(l, mean_l)

        else:
            pass

        return tmp

    def split_data(self, data, dim):
        """Divide the data from highest peak

        Args:
            data: pandas dataframe contain signal(ECG or PPG)
            dim: the dimension for divide

        Returns:
            An ndarrary which looks like [points, dimension]. The points = frequency, and the highest peak always at the middle.
        """
        ignore = 2*self.frequency # avoid the data is start by the high peak
        tmp = pd.DataFrame(data)

        print(tmp)

        ppg_thr = tmp[0].quantile(0.9)
        ecg_thr = tmp[1].quantile(0.98)

        # find the location of R peak
        ppg_tmp = data[:, 0].T
        ecg_tmp = data[:, 1].T

        tmp0 = np.sort(ppg_tmp)
        tmp1 = np.sort(ecg_tmp)

        arg1 = np.argsort(ppg_tmp)
        arg2 = np.argsort(ecg_tmp)

        s1 = 0
        s2 = 0
        for i in range(0, len(tmp0)):
            if tmp0[i] >= ppg_thr and s1 == 0:
                index_ppg = arg1[i] 
                ppg_peak_tmp = arg1[i:]
                s1 = s1 + 1

            if tmp1[i] >= ecg_thr and s2 == 0:
                index_ecg = arg2[i]
                ecg_peak_tmp = arg2[i:]
                s2 = s2 + 1

            if s1 != 0 and s2 != 0 :
                break

        # choose one point represent the peak
        ppg_peak = []
        ecg_peak = []

        ppg_peak_tmp = np.sort(ppg_peak_tmp)
        ecg_peak_tmp = np.sort(ecg_peak_tmp)

        # print(ppg_peak_tmp)

        for j in range(0, len(ppg_peak_tmp)-1): 
            if ppg_peak_tmp[j+1] - ppg_peak_tmp[j] > 20 and ppg_peak_tmp[j] > ignore:
                ppg_peak.append(ppg_peak_tmp[j])

        for k in range(0, len(ecg_peak_tmp)-1):
            if ecg_peak_tmp[k+1] - ecg_peak_tmp[k] > 20 and ecg_peak_tmp[k] > ignore:
                ecg_peak.append(ecg_peak_tmp[k])

        mid_tmp = []
        for m in range(0, len(ppg_peak)-1):
            mid_tmp.append(ppg_peak[m+1] - ppg_peak[m])

        ppg_mid = int(np.mean(mid_tmp) / 2)
        print(ppg_mid)

        mid_tmp = []
        for m in range(0, len(ecg_peak)-1):
            mid_tmp.append(ecg_peak[m+1] - ecg_peak[m])

        ecg_mid = int(np.mean(mid_tmp) / 2)
        print(ecg_mid)

        # split to PPG and ECG
        ppg = []
        ecg = []

        # splite the signal and resample to 2*frequency Hz.
        for i in ppg_peak:
            ppg_sin = tmp.iloc[i-ppg_mid:i+ppg_mid, 0]
            ppg_sin = self.resample(ppg_sin, 2*self.frequency)
            ppg.append(np.asarray(ppg_sin))
            # break

        for i in ecg_peak:
            ecg_sin = tmp.iloc[i-ecg_mid:i+ecg_mid, 1]
            ecg_sin = self.resample(ecg_sin, 2*self.frequency)
            ecg.append(np.asarray(ecg_sin))
            # break

        # ppg = np.asarray(ppg)
        # print(ppg_sin)

        plt.plot(ppg[10])
        plt.plot(ecg[10])
        plt.show()

        print(np.asarray(ppg).shape)
        print(np.asarray(ecg).shape)

        signal = [ppg, ecg]
        return signal

        # print(ppg[0])  


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

    def normalizer(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data)

    def standardizer(self, data):
        return scale(data)  # zero-score normalization

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

        data = self.normalizer(data)
        tmp = self.split_data(data, dim = dim)    

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
        print(self.nerual_number)
        print(self.input_dim)
        model.add(Dense(self.nerual_number, input_shape=(self.input_dim, ), kernel_initializer='normal', activation='relu'))
        # model.add(Dense(self.nerual_number, kernel_initializer='normal', activation='relu'))
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

def remove_zero_rows(Matrix):
    nonzero_row_indice, _ = Matrix.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)

    return Matrix[unique_nonzero_indice]

if __name__ == '__main__':
    
    dp = DataProcess('bidmc_01_Signals.csv')
    data = dp.data

    signals = dp.process_data_origin(data, 1)

    ppg_train = pd.DataFrame(signals[0][:60]) # (60, 200)
    ecg_train = pd.DataFrame(signals[1][:60])

    print(ppg_train.shape)

    nt = Network(ppg_train, ecg_train, 200, 200)
    m = nt.baseline_model()
    for step in range(1001):
        cost = m.train_on_batch(ppg_train, ecg_train)
        if step % 100 == 0:
            print('train cost: ', cost)

    m.evaluate(ppg_train, ecg_train, batch_size=20)
    w, b = m.layers[0].get_weights()
    print('Weights=', w, '\nbiases=', b)


    Y_pred = m.predict(ppg_train)
    # plt.scatter(ppg_train, ecg_train)
    # plt.plot(ppg_train, Y_pred)
    # plt.show()
    # plt.plot(ecg)
    # plt.plot(ppg)
    Y_pred = remove_zero_rows(Y_pred)
    Y_pred = np.mean(Y_pred, axis=1)
    plt.plot(Y_pred)
    plt.show()

