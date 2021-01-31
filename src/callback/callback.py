import numpy as np
import tensorflow as tf
import rootpath

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, subject=None):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.subject = subject
        self.patience = patience
        self.best_weights = None
        self.not_fit = False

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        criteria = 20
        if np.greater(current, criteria):
            self.not_fit = True
            self.model.stop_training = True
            root = rootpath.detect()
            print('These hyper-parameters are not fit in this data')
            with open(root+'/report/not_fit.log', '+a') as f:
                f.write('{0} \n'.format(self.subject))
                f.close()
        elif np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
