from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, TimeDistributed, BatchNormalization, Conv1D, \
    GlobalMaxPooling1D, concatenate
from tensorflow.keras.models import Model, model_from_json

import numpy as np
import json
import pickle
import tarfile
import os
import shutil

#from ner.layers.ChainCRF import ChainCRF
#from ner.layers.layers import CRF
#from tf2crf import CRF
from helper.layers.CRF import CRF

class BiLSTM(object):
    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 use_words=True,
                 use_char=True,
                 use_char_lstm = False,
                 use_casings=True,
                 casing_dim=None,
                 train_casings=False,
                 use_bert=True,
                 use_elmo=False,
                 use_batchNorm=False,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 char_filter_size=30,
                 char_filter_length=3,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=0,
                 dropout=0.5,
                 embeddings=None,
                 train_word_embeddings=True,
                 use_crf=True,
                 sparseTarget=True):

        self._num_labels = num_labels
        self._word_vocab_size = word_vocab_size
        self._use_words = use_words
        self._use_char = use_char
        self._use_char_lstm = use_char_lstm
        self._use_casings = use_casings
        self._casing_dim = casing_dim
        self._train_casings = train_casings
        self._use_bert = use_bert
        self._use_elmo = use_elmo
        self._use_batchNorm = use_batchNorm
        self._char_vocab_size = char_vocab_size
        self._word_embedding_dim = word_embedding_dim
        self._char_embedding_dim = char_embedding_dim
        self._char_filter_size = char_filter_size
        self._char_filter_length = char_filter_length
        self._word_lstm_size = word_lstm_size
        self._char_lstm_size = char_lstm_size
        self._fc_dim = fc_dim
        self._dropout = dropout
        self._embeddings = embeddings
        self._train_word_embeddings = train_word_embeddings
        self._use_crf = use_crf
        self._sparseTarget = sparseTarget

        self._model = None

    """
    Saves the model as tar.gz to a filename
    """
    def save_model(self, filename):
        with open("model.parameter", 'w') as f:
            params = self._model.to_json()
            json.dump(json.loads(params), f, sort_keys=True, indent=4)
        self._model.save_weights("BiLSTMModel/temporary/") #This is nowadays a directory in TF2.0 :/
        pickle.dump((self._num_labels, self._word_vocab_size,  self._use_words, self._use_char, self._use_char_lstm,
                         self._use_casings, self._casing_dim, self._train_casings, self._use_bert, self._use_elmo,\
                         self._use_batchNorm, self._char_vocab_size, self._word_embedding_dim, self._char_embedding_dim,\
                         self._char_filter_size, self._char_filter_length, self._word_lstm_size, self._char_lstm_size, self._fc_dim,\
                         self._dropout, self._embeddings, self._train_word_embeddings, self._use_crf, self._sparseTarget), open("model.property", "wb" ) )

        ####################<Save all to tar.gz>
        with tarfile.open(filename, "w:gz") as tar:
            tar.add("model.parameter")
            tar.add("BiLSTMModel/")
            tar.add("model.property")
        tar.close()

        ####################<Delete the three temporary files>
        os.remove("model.parameter")
        #os.remove("model.weight")
        shutil.rmtree("BiLSTMModel/")
        os.remove("model.property")

    """
    Loads a model from a tar.gz file
    """
    def load_model(filename):

        tar = tarfile.open(filename, "r:gz")

        num_labels, word_vocab_size, use_words, use_char, use_char_lstm, \
            use_casings, casing_dim, train_casings, use_bert, use_elmo, \
            use_batchNorm, char_vocab_size, word_embedding_dim, char_embedding_dim, \
            char_filter_size, char_filter_length, word_lstm_size, char_lstm_size, fc_dim, \
            dropout, embeddings, train_word_embeddings, use_crf, sparseTarget = \
            pickle.load(tar.extractfile(tar.getmember("model.property")))

        tmpVar = BiLSTM(num_labels, word_vocab_size, use_words, use_char, use_char_lstm,
                        use_casings, casing_dim, train_casings, use_bert, use_elmo,
                        use_batchNorm, char_vocab_size, word_embedding_dim, char_embedding_dim,
                        char_filter_size, char_filter_length, word_lstm_size, char_lstm_size, fc_dim,
                        dropout, embeddings, train_word_embeddings, use_crf, sparseTarget
                        )
        tmpVar._model = model_from_json(tar.extractfile(tar.getmember("model.parameter")).read(), custom_objects={'CRF': CRF})

        #Old code to load model from singe file (TF1.0)
        #tar._extract_member(tar.getmember("model.weight"), "/tmp/model.weight")
        #tmpVar._model.load_weights("/tmp/model.weight")
        #os.remove("/tmp/model.weight")


        #New code to load model from directory (TF 2.0=
        subdir_and_files = [
            tarinfo for tarinfo in tar.getmembers()
            if tarinfo.name.startswith("BiLSTMModel")
        ]
        tar.extractall(members=subdir_and_files, path="/tmp/")
        tmpVar._model.load_weights("/tmp/BiLSTMModel/temporary/")
        shutil.rmtree("/tmp/BiLSTMModel/")


        return tmpVar

    def build(self):
        inputs = []
        concat = []

        #<word embeddings>
        if self._use_words:
            word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
            inputs.append(word_ids)
            if self._embeddings is None:
                word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                            output_dim=self._word_embedding_dim,
                                            mask_zero=True,
                                            trainable=self._train_word_embeddings,
                                            name='word_embedding')(word_ids)
            else:
                word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                            output_dim=self._word_embedding_dim,
                                            mask_zero=True,
                                            weights=[self._embeddings],
                                            trainable=self._train_word_embeddings,
                                            name='word_embedding')(word_ids)
            concat.append(word_embeddings)
            # </word embeddings>

        # <Char embedding>
        if self._use_char:

            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=self._use_char_lstm,
                                        name='char_embedding')(char_ids)

            # Use LSTM for char embeddings from Lample et al., 2016
            if self._use_char_lstm:
                char_embeddings = TimeDistributed(Bidirectional(LSTM(units=self._char_lstm_size, dropout=self._dropout, recurrent_dropout=self._dropout)), name="char_lstm")(char_embeddings)

            # Use CNNs for character embeddings from Ma and Hovy, 2016
            # According to Reimers and Gurevych the performance between LSTM and CNN is similar, but CNN are faster
            else:
                char_embeddings = TimeDistributed(Conv1D(self._char_filter_size, self._char_filter_length, padding='same'), name="char_cnn")(char_embeddings)
                char_embeddings = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(char_embeddings)

            concat.append(char_embeddings)
            #word_embeddings = Concatenate()([word_embeddings, char_embeddings])
        # </Char embedding>

        # <Casings>
        # TODO: I'm not sure if this is the best way to solve casings; using embeddings
        if self._use_casings:
            casing_input = Input(batch_shape=(None, None), dtype='int32', name='casing_input')
            inputs.append(casing_input)
            casing_embeddings = Embedding(input_dim=self._casing_dim, output_dim=self._casing_dim,
                                          weights=[np.identity(self._casing_dim, dtype='float32')],
                                          trainable=self._train_casings, name="casing_embedding")(casing_input)
            concat.append(casing_embeddings)
        # </Casings>

        # <Bert>
        if self._use_bert:
            bert_embeddings = Input(batch_shape=(None, None, 768), dtype='float32', name="bert_embeddings")
            #bla = Dropout(0.5)(bert_embeddings)
            inputs.append(bert_embeddings)
            concat.append(bert_embeddings)
        # </Bert>

        #<elmo>
        if self._use_elmo:
            elmo_embeddings = Input(shape=(None, 1024), dtype='float32', name='elmo_embeddings')
            inputs.append(elmo_embeddings)
            concat.append(elmo_embeddings)
        #</elmo>


        #Build concatenated layer
        if len(concat) >= 2:
            concatenated = concatenate(concat)
        else:
            concatenated = concat[0]

        if self._use_batchNorm:
            concatenated = BatchNormalization()(concatenated)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True, dropout=self._dropout, recurrent_dropout=self._dropout))(concatenated)

        # Intermediate layer, was not really necessary according to my experiments
        if self._fc_dim > 0:
            z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            #Version 1: Using layers.crf
            #crf = CRF(self._num_labels, sparse_target=False)
            #loss = crf.loss_function
            #pred = crf(z)
            #metric = [crf.accuracy]

            #Version 2: Using  chaincrf.chaincrf
            #z = TimeDistributed(Dense(self._num_labels, activation=None),name='hidden_lin_layer')(z)
            #crf = ChainCRF(name='CRF')
            #pred = crf(z)
            #loss = crf.loss
            #metric = None

            # Version 3: https://github.com/xuxingya/tf2crf
            output = Dense(self._num_labels, activation=None)(z)
            crf = CRF(dtype='float32', sparse_target=True)
            pred = crf(output)

            if self._sparseTarget == False:
                print("Sparse Target for CRF not implemented, yet!")

            loss = crf.loss
            metric = crf.accuracy

        else:
            if self._sparseTarget ==True:
                loss = 'sparse_categorical_crossentropy'
            else:
                loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)
            metric = ['accuracy']

        self._model = Model(inputs=inputs, outputs=pred)
        self._model.summary()

        self._model.compile(loss=loss, optimizer='nadam', metrics=metric)
