import pickle
import glob
import numpy as np
import json
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import performance_measure

from ner.helper.myModel import BiLSTM
from ner.helper.parseCorpus import load_data_and_labels, docs2id, docs2chars, docs2casings

#Load configuration
with open('../config.json') as json_file:
    config = json.load(json_file)

#GermEval17
testFile = config['aspectGermEval']['iob']['testFile1']
binaryPath = config['aspectGermEval']['binariesPathTaskD']

#Sm3S
testFile = config['aspectSim3S']['iob']['devFile'] #DevFile
testFile = config['aspectSim3S']['iob']['testFile'] #TestFile
binaryPath = config['aspectSim3S']['binariesPathTaskD']

print("Evaluation for: " +testFile)

#Use generator for raining!!!!!!!!!!!!!!
#token2Id,char2Idx,case2Idx,encoder,defaultClass  = pickle.load( open( binaryPath+"GermEval17.pickle", "rb" ) )
#ner = BiLSTM.load_model(binaryPath+"GermEval17.model")

token2Id,char2Idx,case2Idx,encoder,defaultClass  = pickle.load( open( binaryPath+"GermEval17-generator.pickle", "rb" ) )
ner = BiLSTM.load_model(binaryPath+"GermEval17-generator.model")

testTexts, testLabels = load_data_and_labels(testFile)
lengths = list(map(len, testTexts))

testToken = docs2id(testTexts, token2Id=token2Id)
testToken = np.asarray(testToken) #Convert to ndArray
testToken = pad_sequences(testToken, padding='post')
print("Shape of words" + str(testToken.shape) +" - " +str(testToken.dtype))

testCharacters = docs2chars(testTexts, char2Idx=char2Idx)
print("Shape of characters" +str(testCharacters.shape) +" - " +str(testCharacters.dtype))

testCasings = docs2casings(testTexts, case2Idx=case2Idx)
print("Shape of casing" + str(testCasings.shape) +" - " +str(testCasings.dtype))


for i in range(len(testLabels)):
    labels = testLabels[i]
    ###<Check if label is in the training-labels; if not-replace it with 'O'>
    for j in range(len(labels)):
        if labels[j] not in encoder.classes_:
            labels[j] = 'O'

    testLabels[i] = encoder.transform(labels)

testLabels = np.asarray(testLabels) #Convert to ndArraytop
testLabels = pad_sequences(testLabels, padding='post', value=defaultClass)
testLabels = to_categorical(testLabels, len(encoder.classes_)).astype(np.uint8)
print("Shape of labels" + str(testLabels.shape) +" - " +str(testLabels.dtype))

###############Print memory usage :-)
import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key=lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
###</test>

#[testToken, testCharacters, testCasings, testBertEmbeddings]
predicted = ner._model.predict([testToken, testCharacters, testCasings])
y_pred = []
y_true = []
for i in range(predicted.shape[0]):
    pred = predicted[i]
    if ner._use_crf:  # The new CRF implementation returns not probabilities
        y_pred.append(list(encoder.inverse_transform(list(pred.astype(int)))))
    else:
        y_pred.append(list(encoder.inverse_transform(pred.argmax(axis=1))))
    truth = testLabels[i]
    y_true.append(list(encoder.inverse_transform(truth.argmax(axis=1))))

#Trim result to seqeunce length
for i in range(len(lengths)):
    y_pred[i] = y_pred[i][:lengths[i]]
    y_true[i] = y_true[i][:lengths[i]]

print("ACC=" +str(accuracy_score(y_true, y_pred)))
print("F1-micro:%.3f" % f1_score(y_true, y_pred, average="micro"))
print("F1-,acro %.3f" % f1_score(y_true, y_pred, average="macro"))
print(performance_measure(y_true, y_pred))
print(classification_report(y_true, y_pred))

print(binaryPath+"epochModels/")
epochs = [];accuracy = []; f1_micro = []; f1_macro = []; cnt = 0
for file in sorted(glob.glob(binaryPath+"epochModels/*.hdf5")):
    print(file)
    ner._model.load_weights(file)

    predicted = ner._model.predict([testToken, testCharacters, testCasings])
    y_pred = []
    y_true = []
    for i in range(predicted.shape[0]):
        pred = predicted[i]
        if ner._use_crf: #The new CRF implementation returns not probabilities
            y_pred.append(list(encoder.inverse_transform(list(pred.astype(int)))))
        else:
            y_pred.append(list(encoder.inverse_transform(pred.argmax(axis=1))))
        truth = testLabels[i]
        y_true.append(list(encoder.inverse_transform(truth.argmax(axis=1))))

    # Trim result to seqeunce length
    for i in range(len(lengths)):
        y_pred[i] = y_pred[i][:lengths[i]]
        y_true[i] = y_true[i][:lengths[i]]

    cnt = cnt +1
    epochs.append(cnt)
    accuracy.append(accuracy_score(y_true, y_pred))
    f1_micro.append(f1_score(y_true, y_pred, average="micro"))
    f1_macro.append(f1_score(y_true, y_pred, average="macro"))
    print("ACC=%.3f" %accuracy_score(y_true, y_pred))
    print("F1-micro:%.3f" %f1_score(y_true, y_pred, average="micro"))
    print("F1-,acro %.3f" %f1_score(y_true, y_pred, average="macro"))
    print(performance_measure(y_true, y_pred))
#    print(classification_report(y_true, y_pred))
    print("----")

# Plot training & validation accuracy values
import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.plot(epochs,accuracy)
#plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Train-loss', 'Eval-loss'], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(epochs,f1_micro)
#plt.title('F1-Micro')
plt.ylabel('F1-Micro')
plt.xlabel('Epoch')
#plt.legend(['Train-acc', 'Eval-acc'], loc='upper left')

plt.subplot(1, 3, 3)
#plt.plot(history.history['accuracy'],'--')
plt.plot(epochs,f1_macro)
#plt.title('F1-Macro')
plt.ylabel('F1-Macro')
plt.xlabel('Epoch')
#plt.legend(['Train-acc', 'Eval-acc'], loc='upper left')


plt.show()