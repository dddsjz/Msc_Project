#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-24 22:32:43
# @Author  : Junzhe Sun (junzhe.sun13@gmail.com)
# @Link    : http://example.org
# @Version : $Id$

import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
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


    def __init__(self, filename, frequence = 125, minutes = 2):
        current = os.getcwd()
        path = Path(current + '\data')
        file = path / filename
        # print(file)
        self.frequence = frequence
        self.minutes = minutes
        self.data = pd.read_csv(file, usecols=[2, 5], nrows=60*minutes*frequence)  # first min train, second test

    def split_data(self, data, dim=1):
        """Split a pandas dataframe to individual PPG and ECG 

        Args:
            data: a pandas dataframe contains two sperate column PPG(0) & ECG(1)
            dim: divide one slice to 

        Returns:
            A list contain PPG in 0, 1 position and ECG in 2, 3. both ecg and ppg are ndarray
        """
        tmp = []  # return list
        tmp1 = []  # PPG list
        tmp2 = []  # ECG list

        split = int(60*self.frequence*self.minutes/2/dim)

        for i in range(dim):
            tmp1.append(data.iloc[i*split:(i+1)*split, 0])  # PPG train set
            tmp2.append(data.iloc[i*split:(i+1)*split, 1])  # ECG train set

        ppg = np.asarray(tmp1)
        ecg = np.asarray(tmp2)
        # print(ppg.T.shape)
        # print(ecg.shape)

        tmp.append(ppg.T) # x * dim ndarray
        tmp.append(ecg.T)

        # tmp.append(data.iloc[:split, 0])  # PPG train set
        # tmp.append(data.iloc[:split, 1])  # ECG train set
        # tmp.append(data.iloc[split:, 0])  # PPG test set
        # tmp.append(data.iloc[split:, 1])  # ECG test set

        # print(data.iloc[:split, 0].shape)
        # print(data.iloc[split:, 0].shape)
        return tmp 

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
        mean = data.mean()
        std = data.std()
        return data.map(lambda x: (x - mean) / std)  # zero-score normalization

    def normalizer(self, data):
        maximum = data.max()
        minimum = data.min()
        return data.map(lambda x: (x - minimum) / (maximum - minimum))

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

    def inverse_wavelet(self, cA, cD, wavelet='db4', mode='smooth'):
        """
        """
        return waverec(cA, cD, wavelet, mode)

    def ica(self):
        pass

    def process_data_wavelet(self, data):
        """Processing data with wavelet transform

        Args: origin data from xsl file

        Returns: A list include CA4, CD4, CD3, CD2 for both ppg at[0] and ecg at [1]
        """

        data1 = data
        tmp = self.split_data(data1, dim=1)
        ppg = tmp[0]
        ecg = tmp[1]
        wavelet = self.wavelet_transform(ppg)
        # ppg_data = pd.DataFrame(wavelet[0:3])
        ppg_data = wavelet[0:3]
        tmp.append(ppg_data)
        wavelet = self.wavelet_transform(ecg)
        # ecg_data = pd.DataFrame(wavelet[0:3])
        ecg_data = wavelet[0:3]
        tmp.append(ecg_data)

        ppg_test = tmp[2]
        ecg_test = tmp[3]
        wavelet = self.wavelet_transform(ppg_test)
        ppg_data = wavelet[0:3]
        tmp.append(ppg_data)
        wavelet = self.wavelet_transform(ecg_test)
        ecg_data = wavelet[0:3]
        tmp.append(ecg_data)

        return tmp

    def process_data_origin(self, data, dim=10):
        """Processing data with wavelet transform

        Args: 
            data: origin data from xsl file
            dim: the slice number of dividing signal

        Returns: A ndarray for chosen slice size
        """

        data1 = data
        tmp = self.split_data(data1, dim = dim)
        ppg = tmp[0]
        ecg = tmp[1]
        

        return tmp

if __name__ == '__main__':
    obj = DataProcess('bidmc_01_Signals.csv')
    data = obj.data
    tmp = obj.split_data(data)
    ppg = obj.bandpass_filter(tmp[0])
    ecg = obj.bandpass_filter(tmp[1])
    wavelet = obj.wavelet_transform(ppg)
    ac4_ppg, dc4_ppg, dc3_ppg = wavelet[0:3]
    wavelet = obj.wavelet_transform(ecg)
    ac4_ecg, dc4_ecg, dc3_ecg = wavelet[0:3]
    '''
    li = []
    for x in obj.get_median(ecg):
        print(x)

    normal_ppg = obj.normalizer(ppg)
    normal_ecg = obj.normalizer(ecg)
    wavelet = obj.wavelet_transform(ppg)
    ac4_ppg, dc4_ppg, dc3_ppg = wavelet[0:3]
    wavelet = obj.wavelet_transform(ecg)
    ac4_ecg, dc4_ecg, dc3_ecg = wavelet[0:3]
    print(ac4_ecg)
    print(dc3_ecg)
    print(dc4_ecg)
    print(len(ac4_ecg)+len(dc3_ecg)+len(dc4_ecg))
    ecg.plot()
    ppg.plot()
    plt.show()
    '''
    plt.plot(ppg)
    plt.plot(ecg)
    plt.show()
