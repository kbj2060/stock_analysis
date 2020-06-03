import predict
import analysis

class SystemTrading():
	def __init__(self, \
				BATCH_SIZE = 7,\ 
				TIME_STEPS = 15,\
				EPOCH = 10,\
				ITERATIONS = 50,\
				LSTM_UNITS = 64,\
				LEARNING_RATE = 0.0003,\
				DROPOUT_SIZE = 0.3,\
				SUBJECT = '',\
				FEATURES_COUNT = 10,\
				LABEL_COUNT = 1)
		self.BATCH_SIZE = BATCH_SIZE
		self.TIME_STEPS =TIME_STEPS
		self.EPOCH = EPOCH
		self.ITERATIONS = ITERATIONS
		self.LSTM_UNITS = LSTM_UNITS
		self.LEARNING_RATE = LEARNING_RATE
		self.DROPOUT_SIZE = DROPOUT_SIZE
		self.SUBJECT = SUBJECT
		self.FEATURES_COUNT = FEATURES_COUNT
		self.LABEL_COUNT = LABEL_COUNT

