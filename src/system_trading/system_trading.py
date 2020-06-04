from predict import predict
from analysis import analysis

class SystemTrading():
	def __init__(self, SUBJECT, BATCH_SIZE = 7, TIME_STEPS = 15, EPOCH = 10, ITERATIONS = 50, LSTM_UNITS = 64, LEARNING_RATE = 0.0002, DROPOUT_SIZE = 0.4, FEATURES_COUNT = 10, LABEL_COUNT = 1):
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

	def predict(self, options='figure'):
		predict.predict(self.BATCH_SIZE, self.TIME_STEPS, self.SUBJECT, self.LSTM_UNITS, self.FEATURES_COUNT, self.EPOCH, self.ITERATIONS, options)

	def analysis(self):
		return analysis.analysis(self.BATCH_SIZE, self.TIME_STEPS, self.EPOCH, self.ITERATIONS, self.SUBJECT, self.FEATURES_COUNT, self.DROPOUT_SIZE, self.LSTM_UNITS, self.LEARNING_RATE)



	
