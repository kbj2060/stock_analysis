#%%

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np
from keras.optimizers import Adam
import os 
from utils import normalize_dataframe, denormalize, diff_to_percent, create_dataset
from hyperparms import FEATURES_COUNT,LSTM_UNITS,LEARNING_RATE, DROPOUT_SIZE, TRAIN_RATIO, SUBJECT
from dtw import dtw
from math import gcd

def analysis(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS):
    os.chdir('C:\\Users\\kbj20\\OneDrive\\바탕 화면\\trading')
    stock = pd.read_csv('stock\\{s}\\{s}.csv'.format(s=SUBJECT)).drop_duplicates()
    stock.columns = ['Date', 'Close','Open', 'High', 'Low', 'Volume', 'IndividualBuying','ForeignerBuying','InstitutionBuying', 'ForeignerHolding', 'InstitutionHolding', 'Diff','Change' ]
    stock = stock.set_index('Date').dropna()
    stock['Diff'] = [d[-2] * -1 if d[-1] == 'FALL' or d[-1] == 'LOWER_LIMIT' else d[-2] for d in stock.values ]
    stock = stock.drop(['Change'], axis=1)
    stock = stock.sort_index()
    stock['Diff'] = diff_to_percent(stock)
    #stock['Close'] = stock['Close'].ewm(5).mean()
    #print(stock.head(20))
    stock = stock[['Diff','Open','High','Low','Volume', 'IndividualBuying', 'ForeignerBuying', 'InstitutionBuying', 'ForeignerHolding', 'InstitutionHolding','Close']]
    stock = normalize_dataframe(stock)
    stock = stock.fillna(method='ffill')
    #stock = stock[int(len(stock) % ((TIME_STEPS * BATCH_SIZE) / gcd(TIME_STEPS, BATCH_SIZE))):]



    TEST_NUM = BATCH_SIZE * 10 + TIME_STEPS # 250
    TRAIN_NUM = int(len(stock)) - TEST_NUM * 2 # 1596
    temp = TRAIN_NUM - BATCH_SIZE # 1546
    train = stock[temp%BATCH_SIZE:-2*TEST_NUM]
    val = stock[-2*TEST_NUM:-TEST_NUM]
    test = stock[-TEST_NUM:]
    #%%

    # In[7]:

    x_train, y_train = create_dataset(train.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_val, y_val = create_dataset(val.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_test, y_test = create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    print(np.shape(x_train), np.shape(x_val), np.shape(x_test))
    # In[8]:

    x_train = np.reshape(x_train, ( x_train.shape[0], x_train.shape[1], FEATURES_COUNT))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], FEATURES_COUNT))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], FEATURES_COUNT))


    # In[9]:
    model = Sequential()
    model.add(LSTM(LSTM_UNITS,
                batch_input_shape=(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT),
                return_sequences = True
            ))
    model.add(Dropout(DROPOUT_SIZE))
    for i in range(1):
        model.add(LSTM(LSTM_UNITS, batch_input_shape=(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT)))
        model.add(Dropout(DROPOUT_SIZE))
    model.add(Dense(1))

    # 3. 모델 구성하기  batch_input_shape=(1, TIME_STEPS, FEATURES_COUNT)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss = 'mean_squared_error', optimizer = adam, metrics= ['accuracy'])

    # 5. 모델 학습시키기
    for epoch_idx in range(ITERATIONS):
        print('Iterations : ' + str(epoch_idx) + ' / '+str(ITERATIONS) )
        model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=False, validation_data=(x_val, y_val))
        model.reset_states()

    model.save_weights(".\\stock\\{4}\\{4}_bs{3}ts{0}ep{1}it{2}_weight".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT))
    print('Weights Are Saved!')

    model_json = model.to_json()
    with open(".\\stock\\{4}\\{4}_bs{3}ts{0}ep{1}it{2}_model".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT), "w") as json_file:
        json_file.write(model_json)
        print('Model JSON File is saved!')

    #%%

    #x_data, y_data = create_dataset(stock[temp % BATCH_SIZE:].to_numpy(), TIME_STEPS, FEATURES_COUNT)
    pred = model.predict(x_test, batch_size=BATCH_SIZE)

    pred = np.array(pred).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    manhattan_distance = lambda pred, y_test: np.abs(pred - y_test)
    d, cost_matrix, acc_cost_matrix, path = dtw(pred, y_test, dist=manhattan_distance)
    print("Cost is " + str(d))

    plt.figure(figsize=(12, 5))
    plt.plot(pred, 'r', label="prediction")
    plt.plot(y_test, label="answer")
    plt.grid(b=True, which='both', axis='both')
    plt.legend()
    plt.title(str(d))
    plt.savefig('.\\stock\\{3}\\bs{4}ts{0}ep{1}it{2}.png'.format(TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, BATCH_SIZE))
    print("bs{3}ts{0}ep{1}it{2}.png".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE))

    name = "{4}_bs{3}ts{0}ep{1}it{2}".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT)
    result = {name : d}
    print(result)
    return result