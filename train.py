import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
import json
import tensorflow as tf

#from ner.callbacks.callbacks import F1score
from helper.myModel import BiLSTM
from helper.parseCorpus import load_data_and_labels, tokenToId, docs2id, docs2chars, docs2casings, load_glove, \
    filter_embeddings
import matplotlib.pyplot as plt
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


#Load configuration
trainFile = "data/train-2017-09-15.iob"
devFile = "data/devn-2017-09-15.iob"
binaryPath = "binary/"


tf.keras.backend.clear_session()

print("PLEASE USE trainGermEvalGenerator instead!!!! It is much faster and better than this method!")

#Load train data
trainTexts, trainLabels = load_data_and_labels(trainFile, limit=14041)
token2Id = tokenToId(data=trainTexts) #TODO Is there a token which can be mapped to 0? Because 0 is masking


trainToken = docs2id(trainTexts, token2Id=token2Id)
trainToken = np.asarray(trainToken) #Convert to ndArraytop
trainToken = pad_sequences(trainToken, padding='post')
print("Shape of words" + str(trainToken.shape))

encoder = LabelEncoder()
encoder.fit(list(set([item for sublist in trainLabels for item in sublist])))

for i in range(len(trainLabels)):
    label =trainLabels[i]
    trainLabels[i] = encoder.transform(label)

defaultClass = encoder.transform(['O'])[0]
trainLabels = np.asarray(trainLabels) #Convert to ndArraytop
trainLabels = pad_sequences(trainLabels, padding='post', value=defaultClass) #TODO: Is this the correct way to handle this?
#trainLabels = to_categorical(trainLabels, len(encoder.classes_)).astype(int) #Sparse encoding
print("Shape of labels" + str(trainLabels.shape))

char2Idx = {}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx) + 1 #+1 as 0 is masking
trainCharacters = docs2chars(trainTexts, char2Idx=char2Idx)
print("Shape of characters" +str(trainCharacters.shape))


case2Idx = {'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7}
trainCasings = docs2casings(trainTexts, case2Idx=case2Idx)
print("Shape of casings" +str(trainCasings.shape))


embedding_dim=300
embeddings = KeyedVectors.load_word2vec_format(datapath("/home/philippe/workspace/PycharmProjects/ner/embeddings/german.model"), binary=True)

embeddings = filter_embeddings(embeddings, token2Id, embedding_dim)

###<evaluate>
validTexts, validLabels = load_data_and_labels(devFile)
lengths = list(map(len, validTexts))

validToken = docs2id(validTexts, token2Id=token2Id)
validToken = np.asarray(validToken) #Convert to ndArray
validToken = pad_sequences(validToken, padding='post')
print("Shape of words" + str(validToken.shape))

validCharacters = docs2chars(validTexts, char2Idx=char2Idx)
print("Shape of characters" +str(validCharacters.shape))


validCasings = docs2casings(validTexts, case2Idx=case2Idx)
print("Shape of casing" + str(validCasings.shape))


for i in range(len(validLabels)):
    labels =validLabels[i]
    ###<Check if label is in the training-labels; if not-replace it with 'O'>
    for j in range(len(labels)):
        if labels[j] not in encoder.classes_:
            labels[j] = 'O'
    validLabels[i] = encoder.transform(labels)

validLabels = np.asarray(validLabels) #Convert to ndArraytop
validLabels = pad_sequences(validLabels, padding='post', value=defaultClass)
#validLabels = to_categorical(validLabels, len(encoder.classes_)).astype(int) #Sparse encoding
print("Shape of labels" + str(validLabels.shape))
###</evaluate>


ner = BiLSTM(num_labels=len(encoder.classes_),
             use_words = True, train_word_embeddings=True, word_vocab_size=len(token2Id)+2, word_embedding_dim=embedding_dim, ## +2 (0 is padding element; max+1 is unknown element)
             embeddings=embeddings,
             use_char=True, char_vocab_size=len(char2Idx)+2,## +2 (0 is padding element; max+1 is unknown element)
             use_casings=True, casing_dim=len(case2Idx)+1, train_casings=True,
             use_bert=False,
             use_crf=True,
            use_batchNorm=False
             )
bla = ner.build()

plot_model(ner._model, show_shapes=True, to_file=binaryPath +'trainGermEval17.png')


callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, verbose=True))
callbacks.append(EarlyStopping(monitor='val_loss',  patience=5, restore_best_weights=True, verbose=True))
#callbacks = [F1score(encoder)]


history = ner._model.fit(
                         x=[trainToken, trainCharacters, trainCasings],
                         #x=[trainToken, trainCasings],
                         y=trainLabels,
                        validation_data=([validToken, validCharacters, validCasings] ,validLabels),
                        #validation_data=([validToken, validCasings] ,validLabels),
                         epochs=10, batch_size=32,
                        callbacks = callbacks,
                         verbose=1
                         )

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],'--')
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train-loss', 'Eval-loss'], loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Eval-acc'], loc='upper left')

plt.show()

#Save model and all relevant variables
pickle.dump((token2Id,char2Idx,case2Idx,encoder,defaultClass) , open(binaryPath +"GermEval17.pickle", "wb" ))
ner.save_model(binaryPath+"GermEval17.model")

print("PLEASE USE trainGermEvalGenerator instead!!!! It is much faster and better than this method!")
