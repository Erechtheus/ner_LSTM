import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

from helper.generators import NerDataGenerator
from helper.myModel import BiLSTM
from helper.parseCorpus import load_data_and_labels, tokenToId, load_glove, filter_embeddings
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import wandb
from wandb.keras import WandbCallback

keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

trainFile = "data/train.iob"
devFile = "data/test.iob"
binaryPath = "binary/"
embeddingsFile = os.getcwd() +os.sep +"embeddings/german.model"

# 2: Define the search space
projectname="ner-sweep"
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch": {"values": [2, 8, 32, 64, 128]},
        "epochs": {"values": [5, 15, 20, 25]},
        "usenorm" : {"values" : [True, False]},
        "usecrf": {"values": [True, False]},
        "trainEmbeddings": {"values": [True, False]},
        "useCharLSTM" : {"values": [True, False]},

    },
}


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

#token2Id = remove_vocab_withoutEmbedding
# s(embeddings, token2Id)
embeddings = filter_embeddings(embeddings, token2Id, embedding_dim)

def main():
    wandb.init(project=projectname
               )
    print(wandb.config)
    ner = BiLSTM(num_labels=len(encoder.classes_),
                 use_words=True, train_word_embeddings=wandb.config.trainEmbeddings, word_vocab_size=len(token2Id) + 2,
                 word_embedding_dim=embedding_dim,  ## +2 (0 is padding element; max+1 is unknown element)
                 embeddings=embeddings,
                 use_char=True, char_vocab_size=len(char2Idx) + 2, use_char_lstm=wandb.config.useCharLSTM,
                 ## +2 (0 is padding element; max+1 is unknown element)
                 use_casings=True, casing_dim=len(case2Idx) + 1, train_casings=True,
                 use_bert=False,
                 use_crf=wandb.config.usecrf,
                 use_batchNorm=wandb.config.usenorm
                 )
    bla = ner.build()

    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=True))
    callbacks.append(WandbCallback(monitor="val_loss"))

    train_generator = NerDataGenerator(trainTexts, trainLabels, token2Id, encoder, char2Idx, case2Idx, defaultClass=defaultClass, batch_size=wandb.config.batch)#, maxTokensSentences=102, maxCharsToken=198)
    dev_generator = NerDataGenerator(validTexts, validLabels, token2Id, encoder, char2Idx, case2Idx, defaultClass = defaultClass, batch_size=wandb.config.batch)#, maxTokensSentences=154, maxCharsToken=226)

    history = ner._model.fit(x=train_generator,
                            validation_data=dev_generator,
                                epochs=wandb.config.epochs,
                            callbacks=callbacks,
                            verbose=1)



# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=projectname)
wandb.agent(sweep_id, function=main, count=150)
