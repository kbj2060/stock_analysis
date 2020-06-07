#%%

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np
from keras.optimizers import Adam
from src.utils import *
from dtw import dtw
from keras import backend as K
from src.callback import EarlyStoppingAtMinLoss


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

def calc_cost(pred, y_test):
    manhattan_distance = lambda pred, y_test: np.abs(pred - y_test)
    d, cost_matrix, acc_cost_matrix, path = dtw(pred, y_test, dist=manhattan_distance)
    print("Cost is " + str(d))
    return d

def learning(model, x_train, y_train, x_val, y_val, SUBJECT, ITERATIONS, EPOCH, BATCH_SIZE):
    for epoch_idx in range(ITERATIONS):
        print('Iterations : ' + str(epoch_idx) + ' / '+str(ITERATIONS) )
        early_stop = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=False, validation_data=(x_val, y_val), callbacks=[EarlyStoppingAtMinLoss(subject=SUBJECT)])
        print('early stop value is '+str(early_stop))
        model.reset_states()
    return model


def custom_loss(y_true, y_pred):
    b_size = 7
    p_grdt = [ y_pred[i] - y_pred[i-1] for i in range(1, b_size) ]
    t_grdt = [ y_true[i] - y_true[i-1] for i in range(1, b_size) ]
    res = [K.switch(t_grdt[j] * p_grdt[j] >= 0, lambda: K.square(y_true[j] - y_pred[j]), lambda: 2 * K.square(y_true[j] - y_pred[j])) for j in range(0, 6) ]
    return sum(res)/(b_size - 1)

def modeling(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT, DROPOUT_SIZE, LSTM_UNITS, LEARNING_RATE):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS,
                   batch_input_shape=(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT),
                   return_sequences=True
                   ))
    model.add(Dropout(DROPOUT_SIZE))
    for i in range(1):
        model.add(LSTM(LSTM_UNITS, batch_input_shape=(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT)))
        model.add(Dropout(DROPOUT_SIZE))
    model.add(Dense(1))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss=custom_loss, optimizer=adam, metrics=['accuracy'])
    return model

def save_model_weight(model, TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE,SUBJECT, LSTM_UNITS ):
    model.save_weights("stock/{4}/{4}_bs{3}ts{0}ep{1}it{2}lstm{5}_weight".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS))
    print('Weights Are Saved!')
    model_json = model.to_json()
    with open("stock/{4}/{4}_bs{3}ts{0}ep{1}it{2}lstm{5}_model".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS), "w") as json_file:
        json_file.write(model_json)
        print('Model JSON File is saved!')


def analysis(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT, FEATURES_COUNT, DROPOUT_SIZE, LSTM_UNITS, LEARNING_RATE):
    csv = pd.read_csv('stock/{s}/{s}.csv'.format(s=SUBJECT)).drop_duplicates()
    stock = preprocess(csv)
    stock = normalize(stock)
    stock = stock.fillna(method='ffill')

    train, val, test = divide_dataset(stock, BATCH_SIZE, TIME_STEPS)

    x_train, y_train = create_dataset(train.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_val, y_val = create_dataset(val.to_numpy(), TIME_STEPS, FEATURES_COUNT)
    x_test, y_test = create_dataset(test.to_numpy(), TIME_STEPS, FEATURES_COUNT)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], FEATURES_COUNT))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], FEATURES_COUNT))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], FEATURES_COUNT))

    model = modeling(BATCH_SIZE, TIME_STEPS, FEATURES_COUNT, DROPOUT_SIZE, LSTM_UNITS, LEARNING_RATE)
    model = learning(model, x_train, y_train, x_val, y_val, SUBJECT, ITERATIONS, EPOCH, BATCH_SIZE)
    
    save_model_weight(model, TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS)

    pred = model.predict(x_test, batch_size=BATCH_SIZE).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    cost = calc_cost(pred, y_test)
    name = "{4}_bs{3}ts{0}ep{1}it{2}lstm{5}".format(TIME_STEPS, EPOCH, ITERATIONS, BATCH_SIZE, SUBJECT, LSTM_UNITS)
    result = {name : cost}
    return result
