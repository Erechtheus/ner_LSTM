# ner_LSTM
Named Entity Recognition using LSTM/CRF in Keras
This is a rather simple and nowadays old-school implementation of NER using a BiLST and an optional CRF layer.
The project uses rather outdated packages, maybe it works with recent packages as well?

## prerequisite
Download embeddings from https://cloud.devmount.de/d2bc5672c523b086/
Install requirements.

### This works for Python 3.8
conda create -n ner python=3.8
conda activate ner

conda install tensorflow==2.4.3
conda install scikit-learn==0.23.2
conda install gensim==3.8.3
conda install xopen==0.9.0
conda install numpy==1.19.5
pip install tensorflow_addons==0.11.2
pip install wandb==0.12.2

### This works Python 3.9
conda create -n ner2 python=3.9
conda activate ner2

conda install tensorflow-gpu==2.6.2
conda install scikit-learn==0.23.2
conda install gensim
conda install xopen==0.9.0
pip install tensorflow_addons
pip install wandb


##Train your model 
using trainGenerator.py
Please use **this** to train a new model and not train.py! It is much faster than the train.py und leads to better results


## Evaluate your model test.py
Test a NER model  dataset.

## testGenerator.py
Tests a model on the dataset using a generator; but there seems to be some error in the implementation...