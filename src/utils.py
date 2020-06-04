import pandas as pd
import numpy as np
import datetime

def remove_str_from_row(string, row, cols):
    for col in cols:
        row[col] = float(row[col].replace(string, ''))
    return row


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


def divide_dataset(stock, BATCH_SIZE, TIME_STEPS):
    TEST_NUM = BATCH_SIZE * 10 + TIME_STEPS
    TRAIN_NUM = int(len(stock)) - TEST_NUM * 2

    train = stock[(TRAIN_NUM - TIME_STEPS) % BATCH_SIZE:-2 * TEST_NUM]
    val = stock[-2 * TEST_NUM:-TEST_NUM]
    test = stock[-TEST_NUM:]

    return train, val, test

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


def get_ymd(timestamp):
    year = int(str(timestamp)[:4])
    month = int(str(timestamp)[5:7])
    day = int(str(timestamp)[8:10])
    return year, month, day


def batch_workdays(last_day, BATCH_SIZE):
    res, holidays = [], 0
    read_holids = np.array(pd.read_excel('data\\2020_holidays.xlsx')['일자 및 요일'])
    for n in range(1, BATCH_SIZE+1):
        next_day = last_day + datetime.timedelta(days=n+holidays)
        year, month, day = get_ymd(next_day)
        weekday = datetime.date(year, month, day).weekday()
        while weekday == 5 or weekday == 6 or '{0}-{1}-{2}'.format(year, month, day) in read_holids:
            next_day = next_day + datetime.timedelta(days=1)
            year, month, day = get_ymd(next_day)
            weekday = datetime.date(year, month, day).weekday()
            holidays += 1
        res.append(next_day)
    res = [ str(r)[:10] for r in res ]
    return res
