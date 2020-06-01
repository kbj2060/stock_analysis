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
import utils
from dtw import dtw


def analysis(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, FEATURES_COUNT, DROPOUT_SIZE, LSTM_UNITS, LEARNING_RATE):
    csv = pd.read_csv('stock/{s}/{s}.csv'.format(s=SUBJECT)).drop_duplicates()
    stock = utils.preprocess(csv)
    stock = utils.normalize_dataframe(stock)
    stock = stock.fillna(method='ffill')

    TEST_NUM = BATCH_SIZE * 10 + TIME_STEPS # 250
    TRAIN_NUM = int(len(stock)) - TEST_NUM * 2 # 1596

    train = stock[(TRAIN_NUM - BATCH_SIZE)%BATCH_SIZE:-2*TEST_NUM]
    val = stock[-2*TEST_NUM:-TEST_NUM]
    test = stock[-TEST_NUM:]

    x_train, y_train = utils.create_dataset(train.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_val, y_val = utils.create_dataset(val.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_test, y_test = utils.create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], FEATURES_COUNT))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], FEATURES_COUNT))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], FEATURES_COUNT))

    # In[9]:
    # 모델 설계
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

    # 3. 모델 구성
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    # 5. 모델 학습
    for epoch_idx in range(ITERATIONS):
        print('Iterations : ' + str(epoch_idx) + ' / '+str(ITERATIONS) )
        model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=False, validation_data=(x_val, y_val))
        model.reset_states()

    model = utils.save_model_weight(model)
    #%%

    pred = model.predict(x_test, batch_size=BATCH_SIZE).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    manhattan_distance = lambda pred, y_test: np.abs(pred - y_test)
    d, cost_matrix, acc_cost_matrix, path = dtw(pred, y_test, dist=manhattan_distance)
    print("Cost is " + str(d))

    name = "{4}_bs{3}ts{0}ep{1}it{2}lstm{5}".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS)
    result = {name : d}
    print(result)
    return result