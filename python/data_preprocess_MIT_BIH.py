import numpy as np
import wfdb
import pywt
import pandas as pd
import os
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from time2freq_analyze import *
from PreLoader import *
from image_process import *
from LeNet import *
from ResNet import *

# 测试集在数据集中所占的比例
RATIO = 0.2

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data, task_name, sample_num_list):
    number = str(number)
    #ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    ecgClassSet = ['N', 'L', 'R', 'V']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord(f"../data/{task_name}/{number}/{number}", channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(f"../data/{task_name}/{number}/{number}", 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    max_sample_num = 1000
    while i < j:
        try:
            label = ecgClassSet.index(Rclass[i])
            sample_num_list[label] += 1
            if sample_num_list[label] <= max_sample_num:
                x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
                X_data.append(x_train)
                Y_data.append(label)
            i += 1
        except ValueError:
            i += 1
    # print(sample_num_list)
    return 


# 加载数据集并进行预处理
def loadData(task_name):
    
    numberSet = range(100, 125)
    dataSet = []
    labelSet = []
    sample_num_list = [0, 0, 0, 0]
    for n in numberSet:
        try:
            getDataSet(n, dataSet, labelSet, task_name, sample_num_list)
        except AttributeError as error:
            print(f"Pulse dataset-{n} should be skipped because it has problem below: \n{error}")
        except FileNotFoundError as error:
            print(f"Pulse dataset-{n} cannot be found. \n{error}")

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))
    np.random.shuffle(train_ds)

    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]
    data_num, time_length = X.shape[:2]

    return data_num, time_length, Y, X


    