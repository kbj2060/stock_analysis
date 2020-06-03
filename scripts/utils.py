import pandas as pd
import numpy as np
from keras.models import model_from_json
import datetime
from keras.models import load_model

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

def get_model_weight(BATCH_SIZE,TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS):
    json_file = open('/stock/{4}/{4}_bs{0}ts{1}ep{2}it{3}lstm{5}_model'.format(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS), 'r')
    model = json_file.read()
    model = model_from_json(model)
    model.load_weights('/stock/{4}/{4}_bs{0}ts{1}ep{2}it{3}lstm{5}_weight'.format(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS))
    return model

def save_model_weight(model, TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT ):
    model.save_weights("stock/{4}/{4}_bs{3}ts{0}ep{1}it{2}_weight".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT))
    print('Weights Are Saved!')

    model_json = model.to_json()
    with open("stock/{4}/{4}_bs{3}ts{0}ep{1}it{2}_model".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT), "w") as json_file:
        json_file.write(model_json)
        print('Model JSON File is saved!')
    return model

def preprocess(stock):
    stock.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'IndividualBuying', 'ForeignerBuying',
                     'InstitutionBuying', 'ForeignerHolding', 'InstitutionHolding', 'Diff', 'Change']
    stock = stock.set_index('Date').dropna()
    stock['Diff'] = [d[-2] * -1 if d[-1] == 'FALL' or d[-1] == 'LOWER_LIMIT' else d[-2] for d in stock.values]
    stock = stock.drop(['Change'], axis=1)
    stock = stock.sort_index()
    stock['Diff'] = diff_to_percent(stock)
    stock = stock[
        ['Diff', 'Open', 'High', 'Low', 'Volume', 'IndividualBuying', 'ForeignerBuying', 'InstitutionBuying',
         'ForeignerHolding', 'InstitutionHolding', 'Close']]
    return stock


def get_ymd(timestamp):
    year = int(str(timestamp)[:4])
    month = int(str(timestamp)[5:7])
    day = int(str(timestamp)[8:10])
    return year, month, day

def batch_workdays(last_day, BATCH_SIZE):
    res, holidays = [], 0
    read_holids = np.array(pd.read_excel('data/2020_holidays.xlsx')['일자 및 요일'])
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