import analysis
import predict
import hyperparms
import os

os.chdir(os.getcwd()+'/../')
res = {}

BATCH_SIZE, TIME_STEPS, FEATURES_COUNT, EPOCH, ITERATIONS, \
LSTM_UNITS, LEARNING_RATE, DROPOUT_SIZE, SUBJECT = \
hyperparms.BATCH_SIZE, hyperparms.TIME_STEPS, \
hyperparms.FEATURES_COUNT, hyperparms.EPOCH, \
hyperparms.ITERATIONS, hyperparms.LSTM_UNITS,\
hyperparms.LEARNING_RATE, hyperparms.DROPOUT_SIZE, \
hyperparms.SUBJECT

#res.update(analysis.analysis(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, SUBJECT,FEATURES_COUNT, DROPOUT_SIZE, LSTM_UNITS, LEARNING_RATE))
predict.predict(BATCH_SIZE, TIME_STEPS, SUBJECT, LSTM_UNITS, FEATURES_COUNT, EPOCH, ITERATIONS)

