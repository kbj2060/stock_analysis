import pandas as pd
import utils
import numpy as np
import plotly.offline as pyo
from keras.models import model_from_json
import datetime
from keras.models import load_model
import plotly.graph_objects as go


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


def normalize(df):
    for column in df:
        window = df[column].values.reshape(-1, 1)
        min_v = min(window)
        normalized_window = np.array([((float(p) / float(min_v)) - 1) if min_v != 0 else 0 for p in window ]).reshape(len(df), 1)
        df[column] = normalized_window
    return df


def denormalize(arr, fit):
    arr = np.array(arr).reshape(1,-1)
    min_v = min(fit)
    denormalized = [ (i+1) * min_v for i in arr]
    return denormalized


def get_model_weight(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS):
    json_file = open('stock\\{4}\\{4}_bs{0}ts{1}ep{2}it{3}lstm{5}_model.h5'.format(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS), 'r')
    model = json_file.read()
    model = model_from_json(model)
    model.load_weights('stock\\{4}\\{4}_bs{0}ts{1}ep{2}it{3}lstm{5}_weight.h5'.format(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS))
    return model

def draw_graph(y_candle, pred_candle):
    data = dict(type='candlestick',
                x=y_candle.index,
                open=y_candle['Open'],
                high=y_candle['High'],
                low=y_candle['Low'],
                close=y_candle['Close'])

    layout = go.Layout(
        xaxis=dict( range=[pred_candle.index[0], pred_candle.index[-1]] )
    )
    data2 = dict(x=pred_candle.index, y=pred_candle.Close, mode='lines')
    fig = dict(data=[data, data2], layout=layout)
    return fig


def predict(BATCH_SIZE, TIME_STEPS, SUBJECT, LSTM_UNITS, FEATURES_COUNT, EPOCH, ITERATIONS):
    csv = pd.read_csv('stock/{s}/{s}.csv'.format(s=SUBJECT)).sort_index()
    stock = preprocess(csv)
    denorm_stock = stock.copy()
    denorm_stock.index = pd.to_datetime(denorm_stock.index)
    stock = normalize(stock)
    stock = stock.fillna(method='ffill')

    TEST_NUM = (BATCH_SIZE * 10 + TIME_STEPS)
    test = stock[-TEST_NUM:]

    cut = (len(stock)-TIME_STEPS) % BATCH_SIZE
    _stock = stock[cut:]
    _denorm_stock = denorm_stock[cut:]

    x_test, y_test = utils.create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_data, y_data = utils.create_dataset(_stock.to_numpy(), TIME_STEPS, FEATURES_COUNT)

    model = get_model_weight(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS)

    pred = model.predict(x_data, batch_size = BATCH_SIZE)
    future = model.predict(x_test[-BATCH_SIZE:], batch_size=BATCH_SIZE)

    _y_hat = np.array(list(stock['Close'][:TIME_STEPS]) + list(pred) + list(future))
    y_hat = [i[0] if isinstance(i, (np.ndarray, np.generic)) else i for i in _y_hat ]
    denorm_pred = np.array(denormalize(y_hat, denorm_stock['Close'])).reshape(-1, 1)
    
    pred_candle = pd.DataFrame(denorm_pred, columns=['Close'])
    y_candle = _denorm_stock.sort_index()
    
    batch_days = utils.batch_workdays(denorm_stock.index[-1], BATCH_SIZE)
    pred_candle.index = y_candle.index.append(pd.to_datetime(batch_days))
    fig = utils.draw_graph(y_candle, pred_candle)
    name = "{4}_bs{3}ts{0}ep{1}it{2}lstm{5}".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS)
    pyo.plot(fig, filename="stock/{0}/{1}.html".format(SUBJECT, name), auto_open=False)
