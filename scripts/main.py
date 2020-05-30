import sys
from analysis import analysis
from predict import predict
from hyperparms import BATCH_SIZE, TIME_STEPS, FEATURES_COUNT, EPOCH, ITERATIONS, LSTM_UNITS, LEARNING_RATE, DROPOUT_SIZE, SUBJECT

res = {}

for b in BATCH_SIZE:
    # analysis returns cost value
    res.update(analysis(b, TIME_STEPS, EPOCH, ITERATIONS))
    #predict(b, TIME_STEPS)

