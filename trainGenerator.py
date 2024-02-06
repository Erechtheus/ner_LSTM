from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

from helper.generators import NerDataGenerator
from helper.myModel import BiLSTM
from helper.parseCorpus import load_data_and_labels, tokenToId, load_glove, filter_embeddings
import matplotlib.pyplot as plt
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import wandb
from wandb.keras import WandbCallback

keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

wandb.init(
    # set the wandb project where this run will be logged
    project="ner",
)


trainFile = "data/train-2017-09-15.iob"
devFile = "data/devn-2017-09-15.iob"
binaryPath = "binary/"
embeddingsFile = "embeddings/german.model"

trainFile = "data/train.iob"
devFile = "data/test.iob"
binaryPath = "binary/"

#Load data
trainTexts, trainLabels = load_data_and_labels(trainFile)
validTexts, validLabels = load_data_and_labels(devFile)

token2Id = tokenToId(data=trainTexts) #TODO Is there a token which can be mapped to 0? Because 0 is masking

encoder = LabelEncoder()
encoder.fit(list(set([item for sublist in trainLabels for item in sublist])))
defaultClass = encoder.transform(['O'])[0]

char2Idx = {}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx) + 1 #+1 as 0 is masking

case2Idx = {'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7}

embedding_dim=300
embeddings = KeyedVectors.load_word2vec_format(datapath(embeddingsFile), binary=True) #https://cloud.devmount.de/d2bc5672c523b086/

#token2Id = remove_vocab_withoutEmbeddings(embeddings, token2Id)
embeddings = filter_embeddings(embeddings, token2Id, embedding_dim)


ner = BiLSTM(num_labels=len(encoder.classes_),
             use_words = True, train_word_embeddings=True, word_vocab_size=len(token2Id)+2, word_embedding_dim=embedding_dim, ## +2 (0 is padding element; max+1 is unknown element)
             embeddings=embeddings,
             use_char=True, char_vocab_size=len(char2Idx)+2,## +2 (0 is padding element; max+1 is unknown element)
             use_casings=True, casing_dim=len(case2Idx)+1, train_casings=True,
             use_bert=False,
             use_crf=True,
            use_batchNorm=True
             )
bla = ner.build()

plot_model(ner._model, show_shapes=True, to_file=binaryPath +'trainGermEval17-Generator.png')


callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=3, verbose=True))
#callbacks.append(EarlyStopping(monitor='val_loss',  patience=5, restore_best_weights=True, verbose=True))
callbacks.append(ModelCheckpoint(binaryPath+"epochModels/model-weights.{epoch:02d}.hdf5", verbose=1, save_weights_only=True, save_freq='epoch'))#-{val_loss:.2f}
callbacks.append(WandbCallback(monitor="val_loss"))
#callbacks = [F1score(encoder)]

train_generator = NerDataGenerator(trainTexts, trainLabels, token2Id, encoder, char2Idx, case2Idx, defaultClass=defaultClass, batch_size=32)#, maxTokensSentences=102, maxCharsToken=198)
dev_generator = NerDataGenerator(validTexts, validLabels, token2Id, encoder, char2Idx, case2Idx, defaultClass = defaultClass, batch_size=32)#, maxTokensSentences=154, maxCharsToken=226)

history = ner._model.fit(x=train_generator,
                        validation_data=dev_generator,
                            epochs=25,
                        callbacks=callbacks,
                        verbose=1)

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],'--')
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train-loss', 'Eval-loss'], loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'],'--')
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train-acc', 'Eval-acc'], loc='upper left')

plt.show()

#Save model and all relevant variables
pickle.dump((token2Id,char2Idx,case2Idx,encoder,defaultClass) , open(binaryPath +"GermEval17-generator.pickle", "wb" ))
ner.save_model(binaryPath+ "GermEval17-generator.model")