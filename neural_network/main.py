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
        cols_to_use = ['PLETH', 'II']

        self.frequency = frequency
        self.minutes = minutes
        self.points = int(2*60*minutes*frequency)
        self.data = pd.read_csv(file, usecols=cols_to_use, nrows=self.points)[cols_to_use]  # first for train, second for test   

        if resample != 0 and resample_factor != 0:
            tmp0 = self.data
            tmplit = []
            
            i = 0
            while i < self.points:
                tmplit.append(tmp0.iloc[i])
                i = i + resample_factor

            self.data = pd.DataFrame(tmplit)
            self.points = int(self.points / resample_factor)

    def calc(self, data1, data2):
        """
        Work for resample function.
        """
        if data1 > data2:
            return abs(data1 - data2) / 2 + data2
        elif data1 < data2:
            return abs(data2 - data1) / 2 + data1
        else:
            return data1

    def resample(self, signal, aim_feq):
        
        tmp = signal.tolist()


        while(len(tmp) < aim_feq-1):

            rad = random.randint(1, int(len(tmp)/3)-1)
            f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)
            mean_f = self.calc(tmp[f], tmp[f-1])
            mean_m = self.calc(tmp[m], tmp[m-1])
            mean_l = self.calc(tmp[l], tmp[l-1])

            tmp.insert(f, mean_f)
            tmp.insert(m+1, mean_m)
            tmp.insert(l+2, mean_l)

        if len(tmp) < aim_feq:

            rad = random.randint(1, int(len(tmp)/3)-1)
            f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)

            if abs(len(tmp)-aim_feq) % 3 == 1:

                mean_m = self.calc(tmp[m], tmp[m-1])

                tmp.insert(m, mean_m)

            elif abs(len(tmp)-aim_feq) % 3 == 2:

                mean_f = self.calc(tmp[f], tmp[f-1])
                mean_l = self.calc(tmp[l], tmp[l-1])

                tmp.insert(f, mean_f)
                tmp.insert(l+1, mean_l)

            else:
                pass

        else:

            while(len(tmp) - aim_feq >= 3):
                rad = random.randint(1, int(len(tmp)/3)-1)
                f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)

                tmp.pop(l)
                tmp.pop(m)
                tmp.pop(f)

            if len(tmp) > aim_feq:
                rad = random.randint(1, int(len(tmp)/3)-1)
                f, m, l = rad, int(rad + len(tmp)/3), int(rad + 2*len(tmp)/3)

                if abs(len(tmp)-aim_feq) % 3 == 1:

                    tmp.pop(m)

                elif abs(len(tmp)-aim_feq) % 3 == 2:

                    tmp.pop(l)
                    tmp.pop(f)

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

        # find the location of R peak
        ppg_thr = tmp[0].quantile(0.85)
        ecg_thr = tmp[1].quantile(0.90)
        
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
                ppg_peak_tmp = arg1[i:]
                s1 = s1 +1

            if tmp1[i] >= ecg_thr and s2 == 0:
                ecg_peak_tmp = arg2[i:]
                s2 = s2 + 1

            if s1 != 0 and s2 != 0:
                break
       
        ppg_peak_tmp = np.sort(ppg_peak_tmp)
        ecg_peak_tmp = np.sort(ecg_peak_tmp)

        if len(ppg_peak_tmp) <= 30 or len(ecg_peak_tmp) <= 30:
            tmp.drop(tmp.columns[ppg_peak_tmp[0]-50:ppg_peak_tmp[0]+50], axis=1, implace=True)
            return self.split_data(tmp)

        # choose one point represent the peak
        ppg_peak = []
        ecg_peak = []

        for j in range(0, len(ppg_peak_tmp)-1): 
            if ppg_peak_tmp[j+1] - ppg_peak_tmp[j] > 10 and ppg_peak_tmp[j] > ignore:
                ppg_peak.append(ppg_peak_tmp[j])

        for k in range(0, len(ecg_peak_tmp)-1):
            if ecg_peak_tmp[k+1] - ecg_peak_tmp[k] > 10 and ecg_peak_tmp[k] > ignore:
                ecg_peak.append(ecg_peak_tmp[k])

        mid_tmp = []
        for m in range(0, len(ppg_peak)-1):
            mid_tmp.append(ppg_peak[m+1] - ppg_peak[m])

        ppg_mid = int(np.mean(mid_tmp) / 2)

        mid_tmp = []
        for n in range(0, len(ecg_peak)-1):
            mid_tmp.append(ecg_peak[n+1] - ecg_peak[n])

        ecg_mid = int(np.mean(mid_tmp) / 2)

        # split to PPG and ECG
        ppg = []
        ecg = []
        tmp2 = []

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

        signal = [ppg, ecg]
        return signal


    def bandpass_filter(self, signal, order=6, low_feq=0.5, high_feq=45, signal_feq=100):
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
        y = lfilter(b, a, signal)
        return y

    def normalizer(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data)

    def standardizer(self, data):
        return scale(data)  # zero-score normalization

    def process_data_origin(self, data, dim=10):
        """Processing data with wavelet transform

        Args: 
            data: origin data from xsl file
            dim: the slice number of dividing signal

        Returns: A ndarray for chosen slice size
        """
        tmp = np.asarray(data.T)
        tmp[0] = self.bandpass_filter(tmp[0], low_feq = 2, high_feq = 44)
        tmp[1] = self.bandpass_filter(tmp[1], low_feq = 2, high_feq = 44)

        data = (pd.DataFrame(tmp)).T
        # plt.plot(tmp[1][200:2000])
        # plt.plot(tmp[0][200:2000])
        # plt.show()

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
        model.add(Dense(self.nerual_number, input_shape=(self.input_dim, ), kernel_initializer='normal', activation='relu'))
        model.add(Dense(self.nerual_number, kernel_initializer='normal', activation='relu'))
        model.add(Dense(self.nerual_number, kernel_initializer='normal', activation='relu'))

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

def run(mode = 'm', subject = 1):
    
    if subject <= 0 or subject > 25 or (mode != 's' and mode != 'm'):
        print("Subject number should in range 1-25, and mode can only be singal(s) or multi(m)")
        exit(1)

    if mode == 'm':
        dp = DataProcess('0.csv')
        data = dp.data

        thr = 0.024

        signals = dp.process_data_origin(data, 1)

        ppg_tmp = pd.DataFrame(signals[0][:150])
        ecg_tmp = pd.DataFrame(signals[1][:150]) 

        for i in range(1, 25):
            dp = DataProcess(str(i)+".csv")
            data = dp.data
            signals = dp.process_data_origin(data, 1)
            ppg_tmp = pd.concat([ppg_tmp, pd.DataFrame(signals[0][:90])], ignore_index=True, sort=False)
            ecg_tmp = pd.concat([ecg_tmp, pd.DataFrame(signals[1][:90])], ignore_index=True, sort=False)
    else:
        dp = DataProcess(str(subject-1) + '.csv')
        data = dp.data

        thr = 0.037

        signals = dp.process_data_origin(data, 1)

        ppg_tmp = pd.DataFrame(signals[0][:150])
        ecg_tmp = pd.DataFrame(signals[1][:150]) 


    ppgs = pd.DataFrame(ppg_tmp)
    ecgs = pd.DataFrame(ecg_tmp)

    ppg_train = ppgs[:100]
    ecg_train = ecgs[:100]

    ppg_test = ppgs[100:150]
    ecg_test = ecgs[100:150]


    nt = Network(ppg_train, ecg_train, 200, 200)
    m = nt.baseline_model()

    """
    for step in range(1001):
        cost = m.train_on_batch(ppg_train, ecg_train)
        if step % 100 == 0:
            print('train cost: ', cost)
    """
    step = 0
    cost = 1
    while cost > thr:
        cost = m.train_on_batch(ppg_train, ecg_train)
        step = step + 1
        if step % 100 == 0:
            print('train cost: ', cost)

        if step > 3000:
            break

    m.evaluate(ppg_train, ecg_train, batch_size=10)
    w, b = m.layers[0].get_weights()
    print('Weights=', w, '\nbiases=', b)

    Y_pred = m.predict(ppg_test)
    a = Y_pred
    Y_pred = np.mean(Y_pred, axis=0)

    # remove 0s
    Y_pred = Y_pred.tolist()
    x = 0
    length = len(Y_pred)
    while x < length:
        if Y_pred[x] < 0.2:
            Y_pred.pop(x)
            length = length - 1
            x = x - 1
        else:
            x = x + 1

    # resample to 200
    # Y_pred = np.asarray(Y_pred)
    # Y_pred = dp.resample(Y_pred, 200)

    X_test = np.mean(ecg_test, axis=0)

    plt.plot(Y_pred, color = 'r')
    plt.plot(X_test.T, color = 'b')
    plt.show()

    plt.subplot(2, 2, 1)
    plt.plot((a[5]), color = 'r')
    plt.plot((ecg_train.T)[5], color = 'b')

    plt.subplot(2, 2, 2)
    plt.plot((a[41]), color = 'r')
    plt.plot((ecg_train.T)[41], color = 'b')

    plt.subplot(2, 2, 3)
    plt.plot((a[28]), color = 'r')
    plt.plot((ecg_train.T)[28], color = 'b')

    plt.subplot(2, 2, 4)
    plt.plot((a[19]), color = 'r')
    plt.plot((ecg_train.T)[19], color = 'b')

    plt.show()

    return (Y_pred)

if __name__ == '__main__':
    result = run(mode = 's', subject = 22)
    result = run(mode = 'm')