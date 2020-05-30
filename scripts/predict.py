from keras.models import model_from_json
import os
from hyperparms import BATCH_SIZE,TIME_STEPS,FEATURES_COUNT,EPOCH,ITERATIONS,LSTM_UNITS,LEARNING_RATE,DROPOUT_SIZE,SUBJECT
import pandas as pd
from utils import normalize_dataframe, denormalize, diff_to_percent, create_dataset, adjust_index, remove_str_from_row
import matplotlib.pyplot as plt
import numpy as np

def predict(BATCH_SIZE, TIME_STEPS):
    os.chdir('C:\\Users\\kbj20\\OneDrive\\바탕 화면\\trading')

    stock = pd.read_csv('stock\\1\\{s}.csv'.format(s=SUBJECT))
    stock.columns = ['Date', 'Close','Open', 'High', 'Low', 'Volume', 'IndividualBuying','ForeignerBuying','InstitutionBuying', 'ForeignerHolding', 'InstitutionHolding', 'Diff','Change' ]
    stock = stock.set_index('Date').dropna()
    _stock = stock.copy()
    stock['Diff'] = [d[-2] * -1 if d[-1] == 'FALL' or d[-1] == 'LOWER_LIMIT' else d[-2] for d in stock.values ]
    stock = stock.drop(['Change'], axis=1)
    stock = stock.sort_index()
    stock['Diff'] = diff_to_percent(stock)
    #stock['Close'] = stock['Close'].ewm(5).mean()
    stock = stock[['Diff','Open','High','Low','Volume', 'IndividualBuying', 'ForeignerBuying', 'InstitutionBuying', 'ForeignerHolding', 'InstitutionHolding','Close']]
    stock = normalize_dataframe(stock)
    stock = stock.fillna(method='ffill')

    #%%

    TEST_NUM = BATCH_SIZE * 10 + TIME_STEPS  # 250
    TRAIN_NUM = int(len(stock)) - TEST_NUM * 2  # 1596
    temp = TRAIN_NUM - BATCH_SIZE  # 1546
    #print(TEST_NUM, TRAIN_NUM, temp)

    train = stock[temp % BATCH_SIZE:-2 * TEST_NUM]
    val = stock[-2 * TEST_NUM:-TEST_NUM]
    test = stock[-TEST_NUM:]
    #print(len(train), len(val), len(test))

    #%%

    #x_train, y_train = create_dataset(train.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    #x_val, y_val = create_dataset(val.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_test, y_test = create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_data, y_data = create_dataset(stock[temp % BATCH_SIZE:].to_numpy(), TIME_STEPS, FEATURES_COUNT)

    json_file = open('stock\\1\\1_bs{0}ts{1}ep{2}it{3}_model'.format(BATCH_SIZE,TIME_STEPS, EPOCH, ITERATIONS), 'r')
    model = json_file.read()
    model = model_from_json(model)
    model.load_weights('stock\\1\\1_bs{0}ts{1}ep{2}it{3}_weight'.format(BATCH_SIZE,TIME_STEPS, EPOCH, ITERATIONS))
    print(model.summary())


    pred = model.predict(x_data, batch_size=BATCH_SIZE)
    future = model.predict(x_test[-TIME_STEPS:], batch_size=BATCH_SIZE)
    print(denormalize(future, _stock['Close']))

    plt.figure(figsize=(12, 5))
    plt.plot(list(pred)+list(future), 'r', label="prediction")
    plt.plot(list(y_data), label="answer")
    plt.grid(b=True, which='both', axis='both')
    plt.legend()
    plt.show()
    plt.close()

