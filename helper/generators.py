import tensorflow
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from helper.parseCorpus import  docs2id, docs2chars, docs2casings


class NerDataGenerator(tensorflow.keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, instances, labels, token2Id, labelEncoder, char2Idx, case2Idx, defaultClass, batch_size=32,
                 shuffle=True, maxTokensSentences=None, maxCharsToken=None, debug=False):
        'Initialization'
        self.instances = instances
        self.labels = labels
        self.token2Id = token2Id
        self.labelEncoder = labelEncoder
        self.char2Idx = char2Idx
        self.case2Idx = case2Idx
        self.defaultClass = defaultClass
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxTokensSentences = maxTokensSentences
        self.maxCharsToken = maxCharsToken
        self.debug = debug

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.instances) / self.batch_size))

    def __getitem__(self, index):
        if self.debug == True:
            print("index=" +str(index))
        instances = self.instances[index * self.batch_size : min((index + 1) * self.batch_size,len(self.instances))] #min helps us to process all instances, and not only a multiple of batch-size
        blavalidLabels = self.labels[index * self.batch_size : min((index + 1) * self.batch_size, len(self.labels))]

        if self.maxTokensSentences == None:
            maxTokensSentences = max([len(x) for x in instances])  # Max token per sentence
        else:
            maxTokensSentences = self.maxTokensSentences

        if self.maxCharsToken == None:
            maxCharsToken = max([max([len(y) for y in x]) for x in instances])  # Max chars per token
        else:
            maxCharsToken = self.maxCharsToken

        #Do the magic
        tokens = docs2id(instances, token2Id=self.token2Id)
        tokens = np.asarray(tokens)  # Convert to ndArray #TODO make conversion uint and also uint size dependant from token2Id
        tokens = pad_sequences(tokens, padding='post', maxlen=maxTokensSentences)
        if self.debug == True:
            print("Shape of words" + str(tokens.shape))

        characters = docs2chars(instances, char2Idx=self.char2Idx, maxTokensSentences=maxTokensSentences, maxCharsToken=maxCharsToken)
        if self.debug == True:
            print("Shape of characters" + str(characters.shape))

        casings = docs2casings(instances, case2Idx=self.case2Idx, maxTokensSentences=maxTokensSentences)
        if self.debug == True:
            print("Shape of casing" + str(casings.shape))

        for i in range(len(blavalidLabels)):
            labels = blavalidLabels[i]
            ###<Check if label is in the training-labels; if not-replace it with 'O'>
            for j in range(len(labels)):
                if labels[j] not in self.labelEncoder.classes_:
                    labels[j] = 'O'
            blavalidLabels[i] = self.labelEncoder.transform(labels)

        blavalidLabels = np.asarray(blavalidLabels)  # Convert to ndArraytop
        blavalidLabels = pad_sequences(blavalidLabels, padding='post', value=self.defaultClass,  maxlen=maxTokensSentences)
        if len(blavalidLabels.shape) != 2:
            print("Should not happen :)" +str(blavalidLabels.shape))
        #blavalidLabels = to_categorical(blavalidLabels, len(self.labelEncoder.classes_)).astype(np.uint8)
        #print(blavalidLabels.shape)
        #Shape wrong in case of only one-token batch --> reshape
        #if len(blavalidLabels.shape) == 2:
        #blavalidLabels = blavalidLabels.reshape((blavalidLabels.shape[0], 1, blavalidLabels.shape[1]))
        if self.debug == True:
            print("Shape of labels" + str(blavalidLabels.shape))
            print("--------------")

        return [tokens, characters, casings], blavalidLabels

    def on_epoch_end(self):
        c = list(zip(self.instances, self.labels))
        random.shuffle(c)

        self.instances, self.labels = zip(*c)
        self.instances = list(self.instances)
        self.labels = list(self.labels)
