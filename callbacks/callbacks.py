"""
Custom callbacks.
"""
import numpy as np
from tensorflow.keras.callbacks import Callback
from seqeval.metrics import f1_score


class F1score(Callback):

    def __init__(self, encoder):
        super(F1score, self).__init__()
        self._encoder = encoder

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    #https: // stackoverflow.com / questions / 47676248 / accessing - validation - data - within - a - custom - callback
    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        print(self.model)
        x_test = self.validation_data[0]
        y_test = self.validation_data[1]

        predicted = self.model.predict(x_test)
        y_pred = []
        y_true = []
        for i in range(predicted.shape[0]):
            pred = predicted[i]
            y_pred.append(list(self._encoder.inverse_transform(pred.argmax(axis=1))))
            truth = y_test[i]
            y_true.append(list(self._encoder.inverse_transform(truth.argmax(axis=1))))
            #TODO Trim is missing

        # Trim result to seqeunce length
        #for i in range(len(lengths)):
        #    y_pred[i] = y_pred[i][:lengths[i]]
        #    y_true[i] = y_true[i][:lengths[i]]

        #print(f1_score(y_true, y_pred))
        logs['f1'] = f1_score(y_true, y_pred)



"""
class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(label_true, label_pred))
        logs['f1'] = score
"""