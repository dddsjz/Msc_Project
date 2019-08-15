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


    def __init__(self, filename, minutes, frequency = 100, cols_to_use = [' PLETH', ' II']):
        current = os.getcwd()
        path = Path(current + '/data')
        file = path / filename

        self.frequency = frequency
        self.minutes = minutes
        self.points = int(2*60*minutes*frequency)
        self.data = pd.read_csv(file, usecols=cols_to_use, nrows=self.points)[cols_to_use]  # first for train, second for test   

    def calc(self, data1, data2):
        """
        Work for resample function.
        """
        if data1 > data2:
            return (data1 - data2) / 2 + data2
        elif data1 < data2:
            return (data2 - data1) / 2 + data1
        else:
            return data1

    def resample(self, signal, aim_feq):
        if isinstance(signal, list) != True:
            tmp = signal.tolist()
        else:
            tmp = signal
        # print(len(tmp))
        # print(aim_feq)
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
                # print(len(tmp))
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

    def sync_data(self, ppg, ecg):
        """

        Args:
            ppg: A list contain the location of PPG peak
            ecg: A list contain the location of ECG peak

        Returns:
            An array contaion the location of PPG peak at [0] and ECG peak at [1] which is synchronous
              and ppg index at [2], ecg index at [3] for getting correct interval.
        """
        tmp = [[], [], [], []]
        drop = 0
        for i in range(0, len(ppg)):
            for j in range(0, len(ecg)):
                # if ecg[j] - ppg[i] <= 20 and ecg[j] - ppg[i] > 0:
                # print(ppg[i])
                # print(ecg[i])
                if ppg[i] - ecg[j] <= (0.5*self.frequency):
                    tmp[0].append(ppg[i])
                    tmp[1].append(ecg[j])
                    tmp[2].append(i)
                    tmp[3].append(j)
                    drop = drop + 1
                    break

        print('the number of droped samples:'+str(abs(len(ppg) - drop)))
        return tmp

    def split_data(self, data, points):
        """Divide the data from highest peak

        Args:
            data: pandas dataframe contain signal(ECG or PPG)
            
        Returns:
            An ndarrary which looks like [points, dimension]. And the shape is depend by the interval of two peaks.
        """
        ignore = 2*self.frequency # avoid the data is start by the high peak
        tmp = pd.DataFrame(data)

        # find the location of R peak
        ppg_thr = tmp[0].quantile(0.85)
        ecg_thr = tmp[1].quantile(0.98)

        # print(ppg_thr)
        # print(ecg_thr)

        # plt.plot(tmp)
        # plt.show()
        # exit(0)
        
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

        # drop Outliers, if some beat is too high.
        if len(ppg_peak_tmp) <= 30 or len(ecg_peak_tmp) <= 30:
            tmp.drop(tmp.columns[ppg_peak_tmp[0]-50:ppg_peak_tmp[0]+50], axis=1, implace=True)
            return self.split_data(tmp)

        # choose one point represent the peak
        ppg_peak = []
        ecg_peak = []
        add = 0

        # print(len(ppg_peak_tmp))
        # print(len(ppg_peak_tmp))
        # np.set_printoptions(threshold=5000)
        # print(ppg_peak_tmp)
        # print(ppg_peak_tmp)

        # plt.plot(ppg_tmp[200:250])
        # plt.plot(ecg_tmp[200:250])
        # plt.show()

        for j in range(0, len(ppg_peak_tmp)-1): 
            if ppg_peak_tmp[j] > ignore:
                if ppg_peak_tmp[j+1] - ppg_peak_tmp[j] <= 10: # location nearby
                    # print(j)
                    # print(ppg_tmp[ppg_peak_tmp[j+1]] - ppg_tmp[ppg_peak_tmp[j]])
                    if ppg_tmp[ppg_peak_tmp[j+1]] - ppg_tmp[ppg_peak_tmp[j]] < 0 and add == 0: # starting decrease
                        ppg_peak.append(ppg_peak_tmp[j])
                        add = 1  # ignore the following location until next beat
                else:  # j+1 belongs to next beat
                    if add == 0:
                        ppg_peak.append(ppg_peak_tmp[j])
                    else:
                        add = 0

        add = 0
        for k in range(0, len(ecg_peak_tmp)-1): 
            if ecg_peak_tmp[k] > ignore:
                if ecg_peak_tmp[k+1] - ecg_peak_tmp[k] <= 10:  # location nearby
                    if ecg_tmp[ecg_peak_tmp[k+1]] - ecg_tmp[ecg_peak_tmp[k]] < 0 and add == 0: # starting decrease
                        ecg_peak.append(ecg_peak_tmp[k])
                        add = 1  # ignore the following location until next beat
                else:  # j+1 belongs to next beat
                    if add == 0:
                        ecg_peak.append(ecg_peak_tmp[k])
                    else:
                        add = 0
            
        # calculate interval
        # plt.plot(ppg_tmp[200:500])
        # plt.plot(ecg_tmp[200:500])
        # plt.show()

        # print(ppg_peak)
        # print(ecg_peak)

        ppg_mid = []
        for m in range(0, len(ppg_peak)-1):
            ppg_mid.append(int((ppg_peak[m+1] - ppg_peak[m]) / 2))
            # print(ppg_peak[m+1])

        ecg_mid = []
        for n in range(0, len(ecg_peak)-1):
            ecg_mid.append(int((ecg_peak[n+1] - ecg_peak[n]) / 2))

        # sync ppg and ecg
        synced = self.sync_data(ppg_peak, ecg_peak)
        ppg_peak = synced[0]
        ecg_peak = synced[1]
        ppg_lab = synced[2]
        ecg_lab = synced[3]

        if len(ppg_peak) == 0 or len(ecg_peak) == 0:
            return 0

        # print(ppg_peak)
        # print(ecg_peak)
        # exit(0)

        # split to PPG and ECG
        ppg = []
        ecg = []
        tmp2 = []
        droped_ppg = []
        droped_ecg = []
        n = 0

        # find the outliers
        for i in ppg_peak:
            if (n+1) >= len(ppg_lab) or ppg_lab[n]+1 >= len(ppg_mid):
                break

            if ppg_lab[n] == 0:
                pass
            else:
                ppg_sin = tmp.iloc[i-ppg_mid[ppg_lab[n]]+8:i+ppg_mid[ppg_lab[n]+1]+14, 0]

                if abs(len(ppg_sin) - self.frequency) >= (0.15*self.frequency):  # drop outliers
                    droped_ppg.append(n)

            n = n + 1

        n = 0
        for i in ecg_peak:
            if (n+1) >= len(ecg_lab) or ecg_lab[n]+1 >= len(ecg_mid):
                break

            if ecg_lab[n] == 0:
                pass
            else:
                ecg_sin = tmp.iloc[i-ecg_mid[ecg_lab[n]]:i+ecg_mid[ecg_lab[n]+1], 1]

                if abs(len(ecg_sin) - self.frequency) >= (0.5*self.frequency):
                    droped_ecg.append(n)

            n = n + 1

        # splite the signal and resample to 2*frequency Hz.
        n = 0
        for i in ppg_peak:
            if (n+1) >= len(ppg_lab) or ppg_lab[n]+1 >= len(ppg_mid):
                break

            if ppg_lab[n] == 0:
                ppg_sin = tmp.iloc[i-30:i+70, 0] # manuly get the first beat, as the left length does not exist
                ppg_sin = self.resample(ppg_sin, points)
                ppg.append(np.asarray(ppg_sin))
            else:
                ppg_sin = tmp.iloc[i-ppg_mid[ppg_lab[n]]+8:i+ppg_mid[ppg_lab[n]+1]+14, 0]

                # print('i='+str(i)+', label='+str(ppg_lab[n]))
                # print(ppg_mid[ppg_lab[n]])
                # print(ppg_mid[ppg_lab[n]+1])

                if len(ppg_sin) == 0:
                    return 0
                elif n in droped_ppg:  # drop outliers
                    pass
                elif n in droped_ecg:  # if ppg is droped, drop the ecg as well
                    pass
                else:
                    # plt.plot(ppg_sin)
                    # plt.show()

                    ppg_sin = self.resample(ppg_sin, points)

                    if len(ppg_sin) != points:
                        return 0

                    ppg.append(np.asarray(ppg_sin))

            
            n = n + 1

        n = 0
        for i in ecg_peak:
            if (n+1) >= len(ecg_lab) or ecg_lab[n]+1 >= len(ecg_mid):
                break

            if ecg_lab[n] == 0:
                ecg_sin = tmp.iloc[i-50:i+50, 1]
                ecg_sin = self.resample(ecg_sin, points)
                ecg.append(np.asarray(ecg_sin))
            else:
                ecg_sin = tmp.iloc[i-ecg_mid[ecg_lab[n]]:i+ecg_mid[ecg_lab[n]+1], 1]

                # print(i)
                # print(ecg_mid[ecg_lab[n]])
                # print(ecg_mid[ecg_lab[n]+1])

                if len(ecg_sin) == 0:
                    return 0
                elif n in droped_ecg:
                    pass
                elif n in droped_ppg:  # if ppg is droped, drop the ecg as well
                    pass
                else:
                    # plt.plot(ecg_sin)
                    ecg_sin = self.resample(ecg_sin, points)

                    if len(ecg_sin) != points:
                        return 0

                    ecg.append(np.asarray(ecg_sin))

            n = n + 1

        signal = [ppg, ecg]
        
        # tmp1 = pd.DataFrame(signal[0])
        # tmp2 = pd.DataFrame(signal[1])
        
        # plt.xlabel('Time')
        # plt.ylabel('Voltage(mV)')

        
        # plt.plot((tmp2.T), color = 'b')
        # plt.plot((tmp1.T), color = 'r')
        # plt.show()
        # exit(0)

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
        # Todo: check function!
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(data)

    def standardizer(self, data):
        return scale(data)  # zero-score normalization

    def process_data_origin(self, data, points):
        """Processing data with wavelet transform

        Args: 
            data: origin data from xsl file
            dim: the slice number of dividing signal

        Returns: A ndarray for chosen slice size
        """
        tmp = np.asarray(data.T)
        tmp[0] = self.bandpass_filter(tmp[0], low_feq = 1, high_feq = 40)
        tmp[1] = self.bandpass_filter(tmp[1], low_feq = 1, high_feq = 40)

        # plt.plot(tmp[0][100:300], color = 'r')
        # plt.plot(tmp[1][100:300], color = 'b')

        data = (pd.DataFrame(tmp)).T
        data = self.normalizer(data)

        # plt.plot((data.T)[0][100:300], color = 'r')
        # plt.plot((data.T)[1][100:300], color = 'b')

        # plt.xlabel('Time')
        # plt.ylabel('Voltage(mV)')
        # plt.show()

        # exit(0)

        tmp = self.split_data(data, points = points)  

        return tmp

class Network:
    """


    """

    def __init__(self, X, Y, nerual_number, input_dim):
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
        model.compile(loss='mean_squared_error', optimizer='adam') # Todo: check optimizer
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
        print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))

def remove_zero(data, thr = 0.2):
    
    if isinstance(data, list) != True:
        data = data.tolist()
    
    x = 0
    length = len(data)
    while x < length:
        if data[x] < thr:
            data.pop(x)
            length = length - 1
            x = x - 1
        else:
            x = x + 1

    return data

def run(mode = 'm', subject = 1, test_sub=-1, points = 200, minutes = 200/60, ratio = 0.7):
    
    if subject <= 0 or subject > 54 or (mode != 's' and mode != 'm') or test_sub > 54:
        print('Subject number and test_sub number should in range 1-54, and mode can only be single(s) or multi(m)')
        exit(1)

    if mode == 'm':  # mulit subject
        dp = DataProcess('bidmc_1_Signals.csv', minutes)
        data = dp.data

        thr = 0.02
        signals = dp.process_data_origin(data, points)
        test_ECGset = []
        test_PPGset = []

        if signals == 0:
                print('drop subject:0')

        ppg_length = len(signals[0])
        ecg_length = len(signals[1])

        if ecg_length < 40:
                print('drop subject:0')
        else:
            tmp = np.asarray((data))[:, 1]
            tmp = pd.Series(tmp)
            tmp = 1000 * tmp  # amplifiler

            z = np.zeros(tmp.shape[0])
            li = [z, z, tmp, z, z, z]
            array = np.asarray(li)
            out = array.T
            current = os.getcwd()
            np.savetxt(current+'/data/txt/biodb/train_'+str(1)+'.txt', out, fmt = '%d', delimiter = ', ')

            ppg_test = pd.DataFrame(signals[0][int(ppg_length*ratio):ppg_length])
            ecg_test = pd.DataFrame(signals[1][int(ecg_length*ratio):ecg_length])
            test_ECGset.append(ecg_test)
            test_PPGset.append(ppg_test)

        num = 1
        for i in range(1, 53):
            dp = DataProcess('bidmc_' + str(i+1) + '_Signals.csv', minutes)
            data = dp.data
            signals = dp.process_data_origin(data, points)

            if signals == 0:
                print('drop subject:'+str(i))
                continue

            ppg_length = len(signals[0])
            ecg_length = len(signals[1])

            if ecg_length < 40:
                print('drop subject:'+str(i))
                continue

            tmp = np.asarray((data))[:, 1]
            tmp = pd.Series(tmp)
            tmp = 1000 * tmp  # amplifiler

            z = np.zeros(tmp.shape[0])
            li = [z, z, tmp, z, z, z]
            array = np.asarray(li)
            out = array.T
            current = os.getcwd()            
            np.savetxt(current+'/data/txt/biodb/train_'+str(num+1)+'.txt', out, fmt = '%d', delimiter = ', ')

            ppg_test = pd.DataFrame(signals[0][int(ppg_length*ratio):ppg_length])
            ecg_test = pd.DataFrame(signals[1][int(ecg_length*ratio):ecg_length])
            test_ECGset.append(ecg_test)
            test_PPGset.append(ppg_test)

            num = num + 1

        # generate train set
        dp = DataProcess('0.csv', minutes, cols_to_use = ['PLETH', 'II'])
        data = dp.data

        thr = 0.02
        signals = dp.process_data_origin(data, points)

        if signals == 0:
                print('drop subject:0')

        ppg_length = len(signals[0])
        ecg_length = len(signals[1])

        if ecg_length < 40:
                print('drop subject:0')
        else:
            ppg_tmp = pd.DataFrame(signals[0][:int(ppg_length*ratio)])
            ecg_tmp = pd.DataFrame(signals[1][:int(ecg_length*ratio)]) 

        num = 1
        for i in range(1, 25):
            dp = DataProcess(str(i)+".csv", minutes, cols_to_use = ['PLETH', 'II'])
            data = dp.data
            signals = dp.process_data_origin(data, points)

            if signals == 0:
                print('drop subject:'+str(i))
                continue

            ppg_length = len(signals[0])
            ecg_length = len(signals[1])

            if ecg_length < 40:
                print('drop subject:'+str(i))
                continue

            num = num + 1

            ppg_tmp = pd.concat([ppg_tmp, pd.DataFrame(signals[0][:int(ppg_length*ratio)])], ignore_index=True, sort=False)
            ecg_tmp = pd.concat([ecg_tmp, pd.DataFrame(signals[1][:int(ecg_length*ratio)])], ignore_index=True, sort=False)

        ppgs = pd.DataFrame(ppg_tmp)
        ecgs = pd.DataFrame(ecg_tmp)

        ppg_train = ppgs
        ecg_train = ecgs

        if test_sub != -1:
            dp1 = DataProcess('bidmc_' +str(test_sub) + '_Signals.csv', minutes)
            data1 = dp1.data
            test_PPGset = []

            signals1 = dp1.process_data_origin(data, points)

            if signals1 == 0:
                print('the sample '+ str(subject) +' is too noisy.')
                return 0

            ppg_length = len(signals1[0])
            ecg_length = len(signals1[1])

            ppg_tmp1 = pd.DataFrame(signals1[0][:ppg_length])
            ecg_tmp1 = pd.DataFrame(signals1[1][:ecg_length]) 

            ppgs1 = pd.DataFrame(ppg_tmp1)
            ecgs1 = pd.DataFrame(ecg_tmp1)

            ppg_test = ppgs1
            ecg_test = ecgs1

            test_PPGset.append(ppg_test)

    elif test_sub == -1:  # single subject same test 
        dp = DataProcess('bidmc_' + str(subject) + '_Signals.csv', minutes)
        data = dp.data
        test_ECGset = []
        test_PPGset = []
        thr = 0.02

        signals = dp.process_data_origin(data, points)

        if signals == 0:
            print('the sample '+ str(subject) +' is too noisy.')
            return 0

        ppg_length = len(signals[0])
        ecg_length = len(signals[1])

        if ecg_length < 40:
            print('the sample '+ str(subject) +' is too noisy.')
            return 0

        tmp = np.asarray((data))[:, 1]
        tmp = pd.Series(tmp)
        tmp = 1000 * tmp  # amplifiler

        z = np.zeros(tmp.shape[0])
        li = [z, z, tmp, z, z, z]
        array = np.asarray(li)
        out = array.T
        current = os.getcwd()            
        # np.savetxt(current+'/data/txt/biodb/train_'+str(subject)+'.txt', out, fmt = '%d', delimiter = ', ')

        ppg_tmp = pd.DataFrame(signals[0][:int(ppg_length*ratio)])
        ecg_tmp = pd.DataFrame(signals[1][:int(ecg_length*ratio)]) 
        ppg_test = pd.DataFrame(signals[0][int(ppg_length*ratio):ppg_length])
        ecg_test = pd.DataFrame(signals[1][int(ecg_length*ratio):ecg_length])
        test_ECGset.append(ecg_test)
        test_PPGset.append(ppg_test)

        ppgs = pd.DataFrame(ppg_tmp)
        ecgs = pd.DataFrame(ecg_tmp)

        ppg_train = ppgs
        ecg_train = ecgs
        ppg_test = pd.DataFrame(ppg_test)
        ecg_test = pd.DataFrame(ecg_test)
    else:  # single subject different test
        dp = DataProcess('bidmc_' + str(subject) + '_Signals.csv', minutes)
        data = dp.data
        thr = 0.02

        signals = dp.process_data_origin(data, points)

        if signals == 0:
            print('the sample '+ str(subject) +' is too noisy.')
            return 0

        ppg_length = len(signals[0])
        ecg_length = len(signals[1])

        if ecg_length < 40:
            print('the sample '+ str(subject) +' is too noisy.')
            return 0 

        tmp = np.asarray((data))[:, 1]
        tmp = pd.Series(tmp)
        tmp = 1000 * tmp  # amplifiler

        z = np.zeros(tmp.shape[0])
        li = [z, z, tmp, z, z, z]
        array = np.asarray(li)
        out = array.T
        current = os.getcwd()            
        # np.savetxt(current+'/data/txt/biodb/train_'+str(subject)+'.txt', out, fmt = '%d', delimiter = ', ')

        ppg_tmp = pd.DataFrame(signals[0][:int(ppg_length*ratio)])
        ecg_tmp = pd.DataFrame(signals[1][:int(ecg_length*ratio)]) 

        ppgs = pd.DataFrame(ppg_tmp)
        ecgs = pd.DataFrame(ecg_tmp)

        ppg_train = ppgs
        ecg_train = ecgs

        dp1 = DataProcess('bidmc_' +str(test_sub) + '_Signals.csv', minutes)
        data1 = dp1.data
        test_ECGset = []
        test_PPGset = []

        signals1 = dp.process_data_origin(data1, points)

        if signals == 0:
            print('the sample '+ str(test_sub) +' is too noisy.')
            return 0

        ppg_length = len(signals[0])
        ecg_length = len(signals[1])

        if ecg_length < 40:
            print('the sample '+ str(test_sub) +' is too noisy.')
            return 0 

        ppg_test = pd.DataFrame(signals[0][int(ppg_length*ratio):ppg_length])
        ecg_test = pd.DataFrame(signals[1][int(ecg_length*ratio):ecg_length])
        test_ECGset.append(ecg_test)
        test_PPGset.append(ppg_test)

    if ppg_train.shape[0] > ecg_train.shape[0]:
        ppg_train = ppg_train[0:ecg_train.shape[0]]
    elif ppg_train.shape[0] < ecg_train.shape[0]:
        ecg_train = ecg_train[0:ppg_train.shape[0]]
    # plt.plot(ppg_train.T)
    # plt.plot(ecg_train.T)
    # plt.show()

    nt = Network(ppg_train, ecg_train, points, points)
    m = nt.baseline_model()

    step = 0
    cost = 1
    while cost > thr:
        cost = m.train_on_batch(ppg_train, ecg_train)
        step = step + 1
        if step % 100 == 0:
            print('train cost: ', cost)

        if step > 20000:
            break

    m.evaluate(ppg_train, ecg_train, batch_size=10)
    w, b = m.layers[0].get_weights()
    print('Weights=', w, '\nbiases=', b)

    gen_ECG = []
    for i in range(0, len(test_PPGset)):
        Y_pred = m.predict(test_PPGset[i])

        # plt.xlabel('Time')
        # plt.ylabel('Voltage(mV)')
        # plt.plot(Y_pred.T)
        # plt.show()

        tmp = Y_pred.flatten()

        tmp = remove_zero(tmp)

        gen_ECG.append(tmp)
        # plt.plot(tmp)
        # plt.show()

        tmp = pd.Series(tmp)
        tmp = 1000 * tmp  # amplifiler

        z = np.zeros(tmp.shape[0])
        li = [z, z, tmp, z, z, z]
        array = np.asarray(li)
        out = array.T

        current = os.getcwd()
        np.savetxt(current+'/data/txt/biodb/test_c4b'+str(i+1)+'.txt', out, fmt = '%d', delimiter = ', ')

    # plot result
    plt.xlabel('Time')
    plt.ylabel('Voltage(mV)')

        
if __name__ == '__main__':
    run(mode = 'm', points = 100, minutes = 300/60, ratio = 0.5)