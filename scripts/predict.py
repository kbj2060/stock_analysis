import pandas as pd
import utils
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo

def predict(BATCH_SIZE, TIME_STEPS, SUBJECT, LSTM_UNITS, FEATURES_COUNT, EPOCH, ITERATIONS):
    csv = pd.read_csv('stock/{s}/{s}.csv'.format(s=SUBJECT)).sort_index()
    stock = utils.preprocess(csv)
    denorm_stock = stock.copy()
    denorm_stock.index = pd.to_datetime(denorm_stock.index)
    stock = utils.normalize_dataframe(stock)
    stock = stock.fillna(method='ffill')

    #%%

    TEST_NUM = (BATCH_SIZE * 10 + TIME_STEPS)
    test = stock[-TEST_NUM:]

    #%%
    # 잘린 데이터
    cut = (len(stock)-TIME_STEPS) % BATCH_SIZE
    _stock = stock[cut:]
    _denorm_stock = denorm_stock[cut:]

    x_test, y_test = utils.create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_data, y_data = utils.create_dataset(_stock.to_numpy(), TIME_STEPS, FEATURES_COUNT)

    model = utils.get_model_weight(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, LSTM_UNITS)

    pred = model.predict(x_data, batch_size = BATCH_SIZE)
    future = model.predict(x_test[-BATCH_SIZE:], batch_size=BATCH_SIZE)

    _y_hat = np.array(list(stock['Close'][:TIME_STEPS]) + list(pred) + list(future))
    y_hat = [i[0] if isinstance(i, (np.ndarray, np.generic)) else i for i in _y_hat ]

    denorm_pred = np.array(utils.denormalize(y_hat, denorm_stock['Close'])).reshape(-1, 1)
    pred_candle = pd.DataFrame(denorm_pred, columns=['Close'])

    batch_days = utils.batch_workdays(denorm_stock.index[-1], BATCH_SIZE)
    y_candle = _denorm_stock.sort_index()
    pred_candle.index = y_candle.index.append(pd.to_datetime(batch_days))

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
    name = "{4}_bs{3}ts{0}ep{1}it{2}lstm{5}".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS)
    pyo.plot(fig, filename="stock/{0}/{1}.html".format(SUBJECT, name), auto_open=False)
