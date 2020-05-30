
import pandas as pd
import numpy as np
import os
from hyperparms import BATCH_SIZE,TIME_STEPS,FEATURES_COUNT,EPOCH,ITERATIONS,LSTM_UNITS,LEARNING_RATE,DROPOUT_SIZE,SUBJECT

def remove_str_from_row(string, row, cols):
    for col in cols:
        row[col] = float(row[col].replace(string, ''))
    return row

def normalize_dataframe(df):
    for column in df:
        window = df[column].values.reshape(-1, 1)
        min_v = min(window)
        normalized_window = np.array([((float(p) / float(min_v)) - 1) if min_v != 0 else 0 for p in window ]).reshape(len(df), 1)
        df[column] = normalized_window
    return df

def diff_to_percent(stock):
    res = []
    diff = stock['Diff'].values.reshape(-1,1)
    close = stock['Close'].values.reshape(-1,1)
    for i in range(len(diff)):
        if i == 0:
            res.append([0])
            continue
        res.append((close[i]-close[i-1])/close[i-1] *100)
    res = np.array(res)
    return res.flatten()

def create_dataset(data, TIME_STEPS, FEATURES_COUNT):
    dataY = []
    dataX = []
    for i in range(len(data)-TIME_STEPS):
        dataX.append(data[i:(i+TIME_STEPS), :FEATURES_COUNT])
        dataY.append(data[i + TIME_STEPS, -1]) # loc of Diff
    return np.array(dataX), np.array(dataY)

def adjust_index(small, big, col_idx, num):
    big.index = [a[:7] for a in big.index]

    for c in small.columns:
        big[c]= small[c]

    for i, s in enumerate(small.index):
        if i == 0:
            start = 0

def denormalize(arr, fit):
    arr = np.array(arr).reshape(1,-1)
    min_v = min(fit)
    denormalized = [ (i+1) * min_v for i in arr]
    return denormalized
