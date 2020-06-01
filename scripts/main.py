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

costs = []
os.chdir(os.getcwd()+'/../')
SUBJECT = os.listdir('./stock/')

for idx, s in enumerate(SUBJECT[1000:]):
    print("{0} / {1} is executing..".format(idx, len(SUBJECT)))
    try:
        costs.append(analysis(BATCH_SIZE, TIME_STEPS, EPOCH, ITERATIONS, s, LSTM_UNITS, FEATURES_COUNT, DROPOUT_SIZE,  LEARNING_RATE))
        predict(BATCH_SIZE, TIME_STEPS, s, LSTM_UNITS, FEATURES_COUNT, EPOCH, ITERATIONS)
    except Exception as e:
        print('ERROR')
        error = '{0} Error. {1}'.format(s, e)
        with open('errors.log', 'a+') as f:
            f.write(error)
            f.close()

with open('costs.json', 'w+') as costJson:
    costJson.write(costs)
    costJson.close()
