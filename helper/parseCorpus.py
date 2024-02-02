import gensim
import numpy as np
from xopen import xopen

def load_data_and_labels(filename, encoding='utf-8', limit=-1):
    sents, labels = [], []
    words, tags = [], []
    with open(filename) as f:
        for line in f:
            if limit > 0 and len(sents)>= limit:
                break
            line = line.rstrip()
            if line != '':

                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
    return sents, labels


#Builds a token2Id dictionary; where each token gets assigned a unique identifier
def tokenToId(data, lower=True):
    token2Id = {}
    for sentences in data:
        for token in sentences:
            if lower == True:
                token = token.lower()
            if token not in token2Id.keys():
                token2Id[token] = len(token2Id) +1 #We start with ID=1, as ID=0 is the special token for padding
    return token2Id

#Converts a single document into tokens
def doc2ids(doc, token2Id):
    return [token2Id.get(token.lower(), len(token2Id) +1) for token in doc] ##TODO Attention lower is hard coded token.lower() if lower == True else token

#Converts documents into tokens
def docs2id(docs, token2Id):
    return [doc2ids(doc, token2Id=token2Id) for doc in docs]


def docs2chars(docs, char2Idx, maxTokensSentences=None, maxCharsToken=None):

    #We create a three dimensional tensor with
    #Number of samples; Max number of tokens; Max number of characters
    nSamples = len(docs)                                            #Number of samples
    if maxTokensSentences == None:
        maxTokensSentences = max([len(x) for x in docs])                #Max token per sentence
    if maxCharsToken == None:
        maxCharsToken = max([max([len(y) for y in x]) for x in docs])     #Max chars per token

    #print("maxTokensSentences= " +str(maxTokensSentences))
    #print("maxCharsToken= " +str(maxCharsToken))

    x = np.zeros((nSamples,
                  maxTokensSentences,
                  maxCharsToken
                  )).astype('uint8')#probably int32 is to large

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            tokenRepresentation = [char2Idx.get(char, len(char2Idx) + 1) for char in token]
            x[i, j, :len(tokenRepresentation)] = tokenRepresentation

    return(x)


def load_glove(file):
    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with xopen(file) as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model

def remove_vocab_withoutEmbeddings(embeddings, vocab):

    delElements = 0
    for k in list(vocab.keys()):
        if k not in embeddings.vocab:
            del vocab[k]
            delElements = delElements + 1
    print("Removed " + str(delElements) + " keep " + str(len(vocab)) + "= " + str(
        100.0 * delElements / (len(vocab) + delElements)))

    counter = 1
    for k in list(vocab.keys()):
        vocab[k] = counter
        counter = counter + 1

    return vocab

def filter_embeddings(embeddings, vocab, dim, printUnknown=False):
    """Loads word vectors in numpy array.
    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.
    Returns:
        numpy array: an array of word embeddings.
    """
    unknownTokens = 0
    if isinstance(embeddings, dict):
        average_vec = np.mean(np.array(list(embeddings.values()), dtype="float32"), axis=0) #Handle unknown tokens Unknown token (https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
        _embeddings = np.zeros([2 + len(vocab), dim])
        for word in vocab:
            if word in embeddings:
                word_idx = vocab[word]
                _embeddings[word_idx] = embeddings[word]
            # Actually it does not really matter if we set the unknown to 0 or average word...; I think we should add some noise to the embeddings
            else:
                unknownTokens = unknownTokens + 1
                if printUnknown == True:
                    print(word)
        #            _embeddings[word_idx] = average_vec

        _embeddings[1 + max(list(vocab.values()))] = average_vec

    elif isinstance(embeddings, gensim.models.keyedvectors.Word2VecKeyedVectors):
        average_vec = np.mean(embeddings.vectors, axis=0)
        std_vec = np.std(embeddings.vectors, axis=0)
        _embeddings = np.zeros([2 + len(vocab), dim])
        for word in vocab:
            try:
                word_idx = vocab[word]
                _embeddings[word_idx] = embeddings.get_vector(word)
            except KeyError as e:
                unknownTokens = unknownTokens + 1
                if printUnknown == True:
                    print(word)
                #_embeddings[word_idx] = average_vec
                #_embeddings[word_idx] = np.random.normal(average_vec, std_vec)#average vec with random noise
        _embeddings[1 + max(list(vocab.values()))] = average_vec


    else:
        return

    print(str(unknownTokens) +"/" +str(len(vocab)) + " unknown Token")
    return _embeddings


def getCasing(token, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in token:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(token))

    if token.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif token.islower():  # All lower case
        casing = 'allLower'
    elif token.isupper():  # All upper case
        casing = 'allUpper'
    elif token[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]

def docs2casings(docs, case2Idx, maxTokensSentences=None):
    nSamples = len(docs)  # Number of samples
    if maxTokensSentences == None:
        maxTokensSentences = max([len(x) for x in docs])  # Max token per sentence

    x = np.zeros((nSamples,
                  maxTokensSentences
                  )).astype('uint8') #TODO: probably int32 is to large

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            tokenRepresentation = getCasing(token, case2Idx)
            x[i, j] = tokenRepresentation

    return(x)

def doc2bertFast(texts, bc):
    trainBert = np.zeros([len(texts), bc.length_limit, 768], dtype="float32")  # Instances, max_sent_len, #vector representation

    for i, text in enumerate(batch(texts)):

        print(text)
        vec = bc.encode(text, is_tokenized=True)
        trainBert[i] = vec
    return trainBert

def batch(iterable, n=32):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def doc2bert(texts, bc):
    trainBert = np.zeros([len(texts), bc.length_limit, 768], dtype="float32")  # Instances, max_sent_len, #vector representation
    for i, text in enumerate(texts):
        vec = bc.encode([text], is_tokenized=True)  ##TODO Batches makes more sense
        trainBert[i] = vec

        if i % 50 == 0:
            print(i)

    return trainBert
