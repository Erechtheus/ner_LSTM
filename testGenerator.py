import pickle
import json
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import performance_measure

from helper.generators import NerDataGenerator
from helper.myModel import BiLSTM
from helper.parseCorpus import load_data_and_labels

#Load configuration
with open('../config.json') as json_file:
    config = json.load(json_file)

#GermEval17
testFile = config['aspectGermEval']['iob']['testFile1']
binaryPath = config['aspectGermEval']['binariesPathTaskD']

#Sm3S
testFile = config['aspectSim3S']['iob']['devFile']
binaryPath = config['aspectSim3S']['binariesPathTaskD']

#token2Id,char2Idx,case2Idx,encoder,defaultClass  = pickle.load( open( binaryPath+"GermEval17.pickle", "rb" ) )
#ner = BiLSTM.load_model(binaryPath+"GermEval17.model")

token2Id,char2Idx,case2Idx,encoder,defaultClass  = pickle.load( open( binaryPath+"GermEval17-generator.pickle", "rb" ) )
ner = BiLSTM.load_model(binaryPath+"GermEval17-generator.model")

testTexts, testLabels = load_data_and_labels(testFile)
lengths = list(map(len, testTexts))


dev_generator = NerDataGenerator(testTexts, testLabels, token2Id, encoder, char2Idx, case2Idx, defaultClass = defaultClass, batch_size=1, debug=True)#, maxTokensSentences=154, maxCharsToken=226)


###</test>
predicted = ner._model.predict(dev_generator)
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

print(accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(performance_measure(y_true, y_pred))
print(classification_report(y_true, y_pred))
