# FNC-SNLI-FEVER-Domain-Adaptation

## Introduction
This code creates models to apply domain adaptation between the FNC, SNLI,
and FEVER datasets. In particular, we believe that the following labels are 
similar and we can use the datasets to help solve each other's tasks.

<pre>
FNC     agree       disagree       discuss   
SNLI    entailment  contradiction  neutral   
FEVER   SUPPORTS    REFUTES    
</pre>

To correlate the datasets, we assumed the following were similar

<pre>
FNC     headline    article  
SNLI    sentence1   sentence2  
FEVER   claim       wikipedia article  
</pre>

In particular, this code applies the domain adaptation model specified in
this paper https://arxiv.org/abs/1409.7495. The paper is specifically 
designed for unsupervised learning, but we use it in a semi-supervised
manner by training on the labeled data we do have as well. 

More about FNC (Fake New Challenge) can be found here: 
- http://www.fakenewschallenge.org/  

More about SNLI (Stanford Natural Language Inference) can be found here:
- https://nlp.stanford.edu/projects/snli/  

More about FEVER can be found here:
- https://arxiv.org/abs/1803.05355  

## Models & Features Supported
This code supports several models listed below:  
- TF based model
- Average embedding based model
- "Relational" embedding model
- CNN based model

For each of these models, TF, average embedding, "relational" embedding,
and convolutional features respectively are extracted from the data 
headline and corresponding article. The "relational" embedding is the 
average embedding feature except that the headline and article
embeddings are combined multiplicatively and additively.

After feature extraction, the model feeds the features into a hidden 
layer. This hidden layer then feeds into the label classifier. If 
domain adaptation is enabled, the hidden layer also feeds into a domain
classifier which allows for adversarial learning as described by the afor
mentioned paper.

There are also several other options available in the code:  
 - Enabling/disabling labels (var.USE_DISCUSS_LABEL, vars.USE_UNRELATED_LABEL)  
 - Training from pretrained model (var.PRETRAINED_MODEL_PATH) 
 - Balancing data labels (var.BALANCE_DATA)  
 - Balancing validation data size (var.EXTRA_SAMPLES_PER_EPOCH)
 - Appending TF features to hidden before classification 
   (var.ADD_FEATURES_TO_LABEL_PRED)  

## How to Run Models
First, download the data necessary for this model. In particular, the
following data is needed. 

 - Pretrained Word2Vec: https://github.com/mmihaltz/word2vec-GoogleNews-vectors 
 - FNC data: https://github.com/FakeNewsChallenge/fnc-1
 - SNLI data: https://nlp.stanford.edu/projects/snli/
 - FEVER data: http://fever.ai/data.html

Then, run process_data.py to create pickled data for use in train_model.
Before running process_data.py, ensure that the var.py file is configured to 
process all of the data that is needed. After running process_data.py, 
pickled data should appear in the specified PICKLE_SAVE_FOLDER along with
a log file and a copy of var.py.

After getting the required pickle data, run the model with train_model.py.
Ensure before running that var.py is configured correctly. In particular,
choose only 1 feature as the primary input for the model. After running 
the model, training will begin and the validation/test results for each model
is printed and saved. Checkpoints for the lowest validation loss models as
well as a copy of var.py is also saved in var.SAVE_FOLDER.

To test the model, you can use test_model.py with the correct parameters in var.py.
If training a hierarchy model for the FNC dataset, one can use test_fnc_four_label.py
in order to test on the FNC dataset. To run a pretrained model on arbitrary input,
use the code in run_saved_model.py.

## Requirements
The following is a superset of the required libraries via pip freeze. The 
code is written for Python 3.

absl-py==0.1.11  
astor==0.6.2  
autopep8==1.4  
bleach==1.5.0  
boto==2.48.0  
boto3==1.6.6  
botocore==1.9.6  
bz2file==0.98  
certifi==2018.1.18  
chardet==3.0.4  
docutils==0.14  
gast==0.2.0  
gensim==3.4.0  
grpcio==1.10.0  
h5py==2.8.0  
html5lib==0.9999999  
idna==2.6  
jmespath==0.9.3  
json-lines==0.3.1  
Keras==2.2.2  
Keras-Applications==1.0.4  
Keras-Preprocessing==1.0.2  
Mako==1.0.7  
Markdown==2.6.11  
MarkupSafe==1.0  
nltk==3.2.5  
numpy==1.14.1  
protobuf==3.5.2  
pycodestyle==2.4.0  
python-dateutil==2.6.1  
PyYAML==3.12  
requests==2.18.4  
s3transfer==0.1.13  
scikit-learn==0.19.1  
scipy==1.0.0  
six==1.11.0  
sklearn==0.0  
smart-open==1.5.6  
tensorboard==1.6.0  
tensorflow-gpu==1.5.0  
tensorflow-tensorboard==1.5.1  
termcolor==1.1.0  
Theano==1.0.1  
urllib3==1.22  
Werkzeug==0.14.1  

## File and Folder Descriptions
**data**: Folder containing original FNC, SNLI, FEVER, and Word2Vec data  
**flip_gradient.py**: File defining the gradient reversal layer used by domain adaptation  
**models**: Folder used to save trained models  
**pickle_data**: Folder used to save pickled data from process_data.py  
**process_data.py**: Used to process raw training data into usable pickle files  
**train_model.py**: Trains models for task prediction from pickled data  
**util.py**: Provides functions for process_data.py and train_model.py  
**var.py**: Contains parameters used to run process_data.py and train_model.py
**scorer.py**: Contains official scorer used for the FNC task.
**fever_stats.py**: Python script to calculate statistics on the FEVER dataset
**snli_stats.py**: Python script to calculate statistics on the SNLI dataset
**fnc_stats.py**: Python script to calculate statistics on the FNC dataset
**test_model.py**: Contains code used to test a pretrained model on the desired test data
**test_fnc_four_label.py**: Contains code used to test a pretrained hierarchy model on FNC test data.
**run_saved_model.py**: Contains code used to run pretrained model on arbitrary input.

