# ner_LSTM
Named Entity Recognition using LSTM/CRF in Keras
This is a rather simple and nowadays old-school implementation of NER using a BiLST and an optional CRF layer.
The project uses rather outdated packages, maybe it works with recent packages as well?

## prerequisite
Download embeddings from https://cloud.devmount.de/d2bc5672c523b086/
Install requirements.

##Train your model 
using trainGenerator.py
Please use **this** to train a new model and not train.py! It is much faster than the train.py und leads to better results


## Evaluate your model test.py
Test a NER model  dataset.

## testGenerator.py
Tests a model on the dataset using a generator; but there seems to be some error in the implementation...