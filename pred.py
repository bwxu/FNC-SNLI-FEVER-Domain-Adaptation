
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import tensorflow as tf
import numpy as np
import os

from gensim.models.keyedvectors import KeyedVectors
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from flip_gradient import flip_gradient 
from util import get_fnc_data, get_snli_data, get_fever_data, get_vectorizers, get_feature_vectors, save_predictions
from util import get_relational_feature_vectors, remove_stop_words, get_average_embeddings, print_model_results, remove_data_with_label

# Data Processing Params
PICKLE_SAVE_FOLDER = "pickle_data/2_label_only_fever_a_d/"
PICKLE_LOG_FILE = PICKLE_SAVE_FOLDER + "log.txt"

# Saving Parameters
MODEL_NAME = "test"
SAVE_FOLDER = "models/" + MODEL_NAME + "/"
PREDICTION_FILE = SAVE_FOLDER + MODEL_NAME + ".csv"
SAVE_MODEL_PATH = SAVE_FOLDER + MODEL_NAME + ".ckpt"
TRAINING_LOG_FILE = SAVE_FOLDER + "training.txt"
if not os.path.exists(SAVE_FOLDER):
    raise Exception("Folder of model name doesn't exist")

#PRETRAINED_MODEL_PATH = "models/snli_pretrain/snli_pretrain.ckpt"
PRETRAINED_MODEL_PATH = None

# Model options
USE_UNRELATED_LABEL = False
USE_DISCUSS_LABEL = False

# Select train and val datasets
USE_FNC_DATA = False
USE_SNLI_DATA = False
USE_FEVER_DATA = True

# Select test dataset
TEST_DATASET = "FEVER"
if TEST_DATASET not in ["FNC", "FEVER", "SNLI"]:
    raise Exception("TEST_DATASET must be FNC, FEVER, or SNLI")

SNLI_TOTAL_SAMPLES_LIMIT = None
ONLY_VECT_FNC = True

BALANCE_LABELS = False

USE_DOMAINS = False

ADD_FEATURES_TO_LABEL_PRED = False

USE_TF_VECTORS = True
USE_RELATIONAL_FEATURE_VECTORS = False
USE_AVG_EMBEDDINGS = False
USE_CNN_FEATURES = True
input_vectors = [USE_TF_VECTORS, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_CNN_FEATURES]  

EPOCHS = 30
TOTAL_EPOCHS = 30
EPOCH_START = 0
VALIDATION_SET_SIZE = 0.2
NUM_MODELS_TO_TRAIN = 5

if not USE_FNC_DATA and not USE_SNLI_DATA and not USE_FEVER_DATA:
    raise Exception("Must choose data to use")

# Number of extra samples used is EXTRA_SAMPLES_PER_EPOCH * FNC_TRAIN_SIZE
EXTRA_SAMPLES_PER_EPOCH = 1

RATIO_LOSS = 0.5

CHECKS_ENABLED = False
if CHECKS_ENABLED:
    if not USE_UNRELATED_LABEL:
        assert "3_label" in PICKLE_SAVE_FOLDER
    if ADD_FEATURES_TO_LABEL_PRED:
        assert "e_" in MODEL_NAME
    if USE_RELATIONAL_FEATURE_VECTORS:
        assert "_r_" in MODEL_NAME
    if USE_AVG_EMBEDDINGS:
        assert "_avg_" in MODEL_NAME
    if SNLI_TOTAL_SAMPLES_LIMIT == 0:
        assert "only_fnc" in PICKLE_SAVE_FOLDER

# CNN feature paramters
EMBEDDING_PATH = "GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM = 300

# File paths
FNC_TRAIN_STANCES = "fnc_data/train_stances.csv"
FNC_TRAIN_BODIES = "fnc_data/train_bodies.csv"
FNC_TEST_STANCES = "fnc_data/competition_test_stances.csv"
FNC_TEST_BODIES = "fnc_data/competition_test_bodies.csv"

SNLI_TRAIN = 'snli_data/snli_1.0_train.jsonl' 
SNLI_VAL = 'snli_data/snli_1.0_dev.jsonl'
SNLI_TEST = 'snli_data/snli_1.0_test.jsonl'

FEVER_TRAIN = "fever_data/train.jsonl"
FEVER_WIKI = "fever_data/wiki-pages"

# Model parameters
rand0 = random.Random(0)
rand1 = random.Random(1)
rand2 = random.Random(2)
MAX_FEATURES = 5000
TARGET_SIZE = 4
DOMAIN_TARGET_SIZE = 3
HIDDEN_SIZE = 100
DOMAIN_HIDDEN_SIZE = 100
LABEL_HIDDEN_SIZE = None
TRAIN_KEEP_PROB = 0.6
L2_ALPHA = 0.01
CLIP_RATIO = 5
BATCH_SIZE = 100
LR_FACTOR = 0.01

# CNN parameters
FILTER_SIZES = [2, 3, 4]
NUM_FILTERS = 128
#CNN_INPUT_LENGTH = 300
CNN_HEADLINE_LENGTH = 50
CNN_BODY_LENGTH = 500

def process_data():
   with open(PICKLE_LOG_FILE, 'w') as f:
        # Save parameters to log file
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS\n")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        ### Extract Data ###

        print("Getting Data...")
        f.write("Getting Data...\n")
        
        # for now only supports when using FNC data
        assert USE_FNC_DATA == True
        LABELS_TO_IGNORE = set()
        
        if not USE_UNRELATED_LABEL:
            LABELS_TO_IGNORE.add(3)
        if not USE_DISCUSS_LABEL:
            LABELS_TO_IGNORE.add(2)
        
        train_headlines, train_bodies, train_labels, train_domains = [], [], [], []
        val_headlines, val_bodies, val_labels, val_domains = [], [], [], []
        train_sizes = {}
        val_sizes = {}

        VAL_SIZE_CAP = None
        
        if USE_FNC_DATA:
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_body_ids_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
            fnc_headlines_test, fnc_bodies_test, fnc_labels_test, _ = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)
            fnc_domains_train = [0 for i in range(len(fnc_headlines_train))]
            fnc_domains_test = [0 for i in range(len(fnc_headlines_test))]

            # Remove unwanted labels determined by USE_UNRELATED_LABEL and USE_DISCUSS_LABEL
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_domains_train, fnc_body_ids_train = \
                    remove_data_with_label(LABELS_TO_IGNORE, fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_domains_train, additional=fnc_body_ids_train)

            fnc_headlines_test, fnc_bodies_test, fnc_labels_test, fnc_domains_test = \
                    remove_data_with_label(LABELS_TO_IGNORE, fnc_headlines_test, fnc_bodies_test, fnc_labels_test, fnc_domains_test) 
            
            # Seperate the train in to train and validation sets such that
            # no body article or headline is in both the train and validation set
            unique_body_ids = list(set(fnc_body_ids_train))
            indices = list(range(len(unique_body_ids)))
            rand0.shuffle(indices)

            train_body_ids = set(unique_body_ids[i] for i in indices[:len(indices) - int(len(indices) * VALIDATION_SET_SIZE)])
            val_body_ids = set(unique_body_ids[i] for i in indices[len(indices) - int(len(indices) * VALIDATION_SET_SIZE):])

            train_indices = set(i for i in range(len(fnc_body_ids_train)) if fnc_body_ids_train[i] in train_body_ids)
            val_indices = set(i for i in range(len(fnc_body_ids_train)) if fnc_body_ids_train[i] in val_body_ids)

            val_headlines += [fnc_headlines_train[i] for i in val_indices]
            val_bodies += [fnc_bodies_train[i] for i in val_indices]
            val_labels += [fnc_labels_train[i] for i in val_indices]
            val_domains += [fnc_domains_train[i] for i in val_indices]
            
            train_headlines = [fnc_headlines_train[i] for i in train_indices]
            train_bodies = [fnc_bodies_train[i] for i in train_indices]
            train_labels = [fnc_labels_train[i] for i in train_indices]
            train_domains = [fnc_domains_train[i] for i in train_indices]

            train_sizes['fnc'] = len(train_indices)
            val_sizes['fnc'] = len(val_indices)
            
            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines) * EXTRA_SAMPLES_PER_EPOCH
 
        if USE_FEVER_DATA:
            fever_headlines, fever_bodies, fever_labels, fever_claim_set = get_fever_data(FEVER_TRAIN, FEVER_WIKI)
            fever_domains = [2 for _ in range(len(fever_headlines))]

            fever_headlines, fever_bodies, fever_labels, fever_domains = \
                remove_data_with_label(LABELS_TO_IGNORE, fever_headlines, fever_bodies, fever_labels, fever_domains)
 
            claim_list = list(fever_claim_set)
            claim_indices = list(range(len(claim_list)))
            rand1.shuffle(claim_indices)

            fever_val_test_size = int(len(claim_indices) * VALIDATION_SET_SIZE)
            train_claims = set([claim_list[i] for i in claim_indices[:len(claim_indices) - 2 *fever_val_test_size]])
            val_claims = set([claim_list[i] for i in claim_indices[len(claim_indices) - 2 * fever_val_test_size:len(claim_indices) - fever_val_test_size]])
            test_claims = set([claim_list[i] for i in claim_indices[len(claim_indices) - fever_val_test_size:]])

            train_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in train_claims]
            val_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in val_claims]
            test_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in test_claims]

            train_headlines += [fever_headlines[i] for i in train_indices]
            train_bodies += [fever_bodies[i] for i in train_indices]
            train_labels += [fever_labels[i] for i in train_indices]
            train_domains += [fever_domains[i] for i in train_indices]

            val_headlines += [fever_headlines[i] for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [fever_bodies[i] for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [fever_labels[i] for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [fever_domains[i] for i in val_indices][:VAL_SIZE_CAP]

            fever_headlines_test = [fever_headlines[i] for i in test_indices]
            fever_bodies_test = [fever_bodies[i] for i in test_indices]
            fever_labels_test = [fever_labels[i] for i in test_indices]
            fever_domains_test = [fever_domains[i] for i in test_indices]

            train_sizes['fever'] = len(train_indices)
            val_sizes['fever'] = len(val_indices)
            
            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines) * EXTRA_SAMPLES_PER_EPOCH
           
        if USE_SNLI_DATA:
            snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=SNLI_TOTAL_SAMPLES_LIMIT)
            snli_domains = [1 for _ in range(len(snli_s1_train))]
            
            snli_s1_train, snli_s2_train, snli_labels_train, snli_domains = \
                remove_data_with_label(LABELS_TO_IGNORE, snli_s1_train, snli_s2_train, snli_labels_train, snli_domains)
            
            s2_list = list(set(snli_s2_train))
            s2_indices = list(range(len(s2_list)))
            rand1.shuffle(s2_indices)

            snli_val_test_size = int(len(s2_indices) * VALIDATION_SET_SIZE)
            train_s2 = set([s2_list[i] for i in s2_indices[:len(s2_indices) - 2 * snli_val_test_size]])
            val_s2 = set([s2_list[i] for i in s2_indices[len(s2_indices) - 2 * snli_val_test_size:len(s2_indices) - snli_val_test_size]])
            test_s2 = set([s2_list[i] for i in s2_indices[len(s2_indices) - snli_val_test_size:]])

            train_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in train_s2]
            val_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in val_s2]
            test_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in test_s2]
            
            train_headlines += [snli_s1_train[i] for i in train_indices]
            train_bodies += [snli_s2_train[i] for i in train_indices]
            train_labels += [snli_labels_train[i] for i in train_indices]
            train_domains += [snli_domains[i] for i in train_indices]

            val_headlines += [snli_s1_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [snli_s2_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [snli_labels_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [snli_domains[i] for i in val_indices][:VAL_SIZE_CAP]
            
            snli_headlines_test = [snli_s1_train[i] for i in test_indices]
            snli_bodies_test = [snli_s2_train[i] for i in test_indices]
            snli_labels_test = [snli_labels_train[i] for i in test_indices]
            snli_domains_test = [snli_domains[i] for i in train_indices]
            
            train_sizes['snli'] = len(train_indices)
            val_sizes['snli'] = len(val_indices)

            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines) * EXTRA_SAMPLES_PER_EPOCH

        np.save(PICKLE_SAVE_FOLDER + "train_sizes.npy", train_sizes)
        np.save(PICKLE_SAVE_FOLDER + "val_sizes.npy", val_sizes)

        np.save(PICKLE_SAVE_FOLDER + "train_labels.npy", np.asarray(train_labels))
        del train_labels
        np.save(PICKLE_SAVE_FOLDER + "train_domains.npy", np.asarray(train_domains))
        del train_domains

        np.save(PICKLE_SAVE_FOLDER + "val_labels.npy", np.asarray(val_labels))
        del val_labels
        np.save(PICKLE_SAVE_FOLDER + "val_domains.npy", np.asarray(val_domains))
        del val_domains
        
        if TEST_DATASET == 'FNC':
            test_headlines = fnc_headlines_test
            test_bodies = fnc_bodies_test
            test_labels = fnc_labels_test
            test_domains = fnc_domains_test

        if TEST_DATASET == 'FEVER':
            test_headlines = fever_headlines_test
            test_bodies = fever_bodies_test
            test_labels = fever_labels_test
            test_domains = fever_domains_test

        if TEST_DATASET == 'FEVER':
            test_headlines = fever_headlines_test
            test_bodies = fever_bodies_test
            test_labels = fever_labels_test
            test_domains = fever_domains_test
           
        test_headlines, test_bodies, test_labels, test_domains = \
            remove_data_with_label(LABELS_TO_IGNORE, test_headlines, test_bodies, test_labels, test_domains)

        np.save(PICKLE_SAVE_FOLDER + "test_labels.npy", np.asarray(test_labels))
        del test_labels
        np.save(PICKLE_SAVE_FOLDER + "test_domains.npy", np.asarray(test_domains))
        del test_domains

        ### Get TF and TFIDF vectorizers ###

        print("Creating Vectorizers...")
        f.write("Creating Vectorizers...\n")

        vec_train_data = train_headlines + train_bodies

        # Only train vectorizers with FNC data
        if ONLY_VECT_FNC and USE_FNC_DATA:
            vec_train_data = train_headlines[:train_sizes['fnc']] + fnc_bodies_train[:train_sizes['fnc']]
       
        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(vec_train_data, MAX_FEATURES)
        del vec_train_data

        ### Get Feature Vectors ###

        if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
            print("Getting Feature Vectors...")
            f.write("Getting Feature Vectors...\n")
            
            print("  train...")
            f.write("  train...\n")
            train_tf_vectors = get_feature_vectors(train_headlines, train_bodies,
                                                   bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(PICKLE_SAVE_FOLDER + "train_tf_vectors.npy", train_tf_vectors)
            f.write("    Number of Feature Vectors: " + str(len(train_tf_vectors)) + "\n")

            print("  val...")
            f.write("  val...\n")
            val_tf_vectors = get_feature_vectors(val_headlines, val_bodies,
                                                 bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(PICKLE_SAVE_FOLDER + "val_tf_vectors.npy", val_tf_vectors)
            f.write("    Number of Feature Vectors: " + str(len(val_tf_vectors)) + "\n")

            print("  test...")
            f.write("  test...\n")
            test_tf_vectors = get_feature_vectors(test_headlines, test_bodies,
                                               bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            
            f.write("    Number of Feature Vectors: " + str(len(test_tf_vectors)) + "\n")
            np.save(PICKLE_SAVE_FOLDER + "test_tf_vectors.npy", test_tf_vectors)
           
        if USE_RELATIONAL_FEATURE_VECTORS:
            print("Getting Relational Feature Vectors...")
            f.write("Getting Relational Feature Vectors...\n")
            
            print("  train...")
            f.write("  train...\n")
            train_relation_vectors = get_relational_feature_vectors(train_tf_vectors)
            del train_tf_vectors
            np.save(PICKLE_SAVE_FOLDER + "train_relation_vectors.npy", train_relation_vectors)
            del train_relation_vectors

            print("  val...")
            f.write("  val...\n")
            val_relation_vectors = get_relational_feature_vectors(val_tf_vectors)
            del val_tf_vectors
            np.save(PICKLE_SAVE_FOLDER + "val_relation_vectors.npy", val_relation_vectors)
            del val_relation_vectors

            print("  test...")
            f.write("  test...\n")
            test_relation_vectors = get_relational_feature_vectors(test_tf_vectors)
            del test_tf_vectors
            test_relation_vectors = np.asarray(test_relation_vectors)
            np.save(PICKLE_SAVE_FOLDER + "test_relation_vectors.npy", test_relation_vectors)
            del test_relation_vectors

        if USE_AVG_EMBEDDINGS or USE_CNN_FEATURES:
            print("Proessing Embedding Data")
            f.write("Processing Embedding Data...\n")

            print("  Getting pretrained embeddings...")
            f.write("  Getting pretrained embeddings...\n")
            embeddings = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=True)

            print("  Removing Stop Words...")
            f.write("  Removing Stop Words...\n") 
            # Remove stopwords from training and test data
            train_headlines = remove_stop_words(train_headlines)
            train_bodies = remove_stop_words(train_bodies)

            val_headlines = remove_stop_words(val_headlines)
            val_bodies = remove_stop_words(val_bodies)
            
            test_headlines = remove_stop_words(test_headlines)
            test_bodies = remove_stop_words(test_bodies)
            
        if USE_AVG_EMBEDDINGS:
            print("Getting Average Embedding Vectors...")
            f.write("Getting Average Embedding Vectors...\n")

            print("  Train Headline...")
            f.write("  Train Headline...\n")
            train_headline_avg_embeddings = get_average_embeddings(train_headlines, embeddings)
 
            print("  Train Body...")
            f.write("  Train Body...\n")
            train_body_avg_embeddings = get_average_embeddings(train_bodies, embeddings)

            print("  Val Headline...")
            f.write("  Val Headline...\n")
            val_headline_avg_embeddings = get_average_embeddings(val_headlines, embeddings)

            print("  Val Body...")
            f.write("  Val Body...\n")
            val_body_avg_embeddings = get_average_embeddings(val_bodies, embeddings)
            
            print("  Test Headline...")
            f.write("  Test Headline...\n") 
            test_headline_avg_embeddings = get_average_embeddings(test_headlines, embeddings)
            
            print("  Train Headline...")
            f.write("  Train Headline...\n") 
            test_body_avg_embeddings = get_average_embeddings(test_bodies, embeddings)
            
            print("  Combining Train Vectors...")
            f.write("  Combining Train Vectors...\n")
            train_avg_embed_vectors = [np.concatenate([train_headline_avg_embeddings[i], train_body_avg_embeddings[i]])
                                       for i in range(len(train_headline_avg_embeddings))]
            np.save(PICKLE_SAVE_FOLDER + "train_avg_embed_vectors.npy", train_avg_embed_vectors)
            del train_avg_embed_vectors
            
            print("  Combining Val Vectors...")
            f.write("  Combining Val Vectors...")
            val_avg_embed_vectors = [np.concatenate([val_headline_avg_embeddings[i], train_body_avg_embeddings[i]])
                                     for i in range(len(val_headline_avg_embeddings))]
            np.save(PICKLE_SAVE_FOLDER + "val_avg_embed_vectors.npy", val_avg_embed_vectors)
            del val_avg_embed_vectors

            print("  Combining Test Vectors...")
            f.write("  Combining Test Vectors...\n")
            test_avg_embed_vectors = [np.concatenate([test_headline_avg_embeddings[i], test_body_avg_embeddings[i]])
                                      for i in range(len(test_headline_avg_embeddings))]
            np.save(PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy", test_avg_embed_vectors)
            del test_avg_embed_vectors
        
        if USE_CNN_FEATURES:
            print("Getting CNN Input Vectors...")
            f.write("Getting CNN Input Vectors...\n")

            print("  Tokenizing Text...")
            f.write("  Tokenizing Text...\n")
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_headlines + train_bodies)

            train_headline_seq = tokenizer.texts_to_sequences(train_headlines)
            train_body_seq = tokenizer.texts_to_sequences(train_bodies)
            
            val_headline_seq = tokenizer.texts_to_sequences(val_headlines)
            val_body_seq = tokenizer.texts_to_sequences(val_bodies)

            test_headline_seq = tokenizer.texts_to_sequences(test_headlines)
            test_body_seq = tokenizer.texts_to_sequences(test_bodies)
            
            word_index = tokenizer.word_index
            np.save(PICKLE_SAVE_FOLDER + "word_index.npy", word_index)

            print("  Padding Sequences...")
            f.write("  Padding Sequences...\n")
            x_train_headlines = pad_sequences(train_headline_seq, maxlen=CNN_HEADLINE_LENGTH)
            x_train_bodies = pad_sequences(train_body_seq, maxlen=CNN_BODY_LENGTH)
            
            x_val_headlines = pad_sequences(val_headline_seq, maxlen=CNN_HEADLINE_LENGTH)
            x_val_bodies = pad_sequences(val_body_seq, maxlen=CNN_BODY_LENGTH)

            x_test_headlines = pad_sequences(test_headline_seq, maxlen=CNN_HEADLINE_LENGTH)
            x_test_bodies = pad_sequences(test_body_seq, maxlen=CNN_BODY_LENGTH)

            np.save(PICKLE_SAVE_FOLDER + "x_train_headlines.npy", x_train_headlines)
            np.save(PICKLE_SAVE_FOLDER + "x_train_bodies.npy", x_train_bodies)

            np.save(PICKLE_SAVE_FOLDER + "x_val_headlines.npy", x_val_headlines)
            np.save(PICKLE_SAVE_FOLDER + "x_val_bodies.npy", x_val_bodies)

            np.save(PICKLE_SAVE_FOLDER + "x_test_headlines.npy", x_test_headlines)
            np.save(PICKLE_SAVE_FOLDER + "x_test_bodies.npy", x_test_bodies)

            print("  Creating Embedding Matrix...")
            f.write(" Creating Embedding Matrix...\n")
            num_words = len(word_index)
            embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
            for word, rank in word_index.items():
                if word in embeddings.vocab:
                    embedding_matrix[rank] = embeddings[word]

            np.save(PICKLE_SAVE_FOLDER + "embedding_matrix.npy", embedding_matrix)


def train_model():
    with open(TRAINING_LOG_FILE, 'w') as f:
        # Loop for training multiple models\
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, \
                USE_AVG_EMBEDDINGS, USE_TF_VECTORS, USE_CNN_FEATURES]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, \
               USE_AVG_EMBEDDINGS, USE_TF_VECTORS, USE_CNN_FEATURES")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, \
                USE_AVG_EMBEDDINGS, USE_TF_VECTORS, USE_CNN_FEATURES\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        # Take last VALIDATION_SET_SIZE PROPORTION of train set as validation set
        print("Loading train vectors...")
        f.write("Loading train vectors...\n")
        train_sizes = np.load(PICKLE_SAVE_FOLDER + "train_sizes.npy")
        train_sizes = train_sizes.item()
        train_labels = np.load(PICKLE_SAVE_FOLDER + "train_labels.npy")
        train_domains = np.load(PICKLE_SAVE_FOLDER + "train_domains.npy")
        
        print("Loading val vectors...")
        f.write("Loading val vectors...\n")
        val_sizes = np.load(PICKLE_SAVE_FOLDER + "val_sizes.npy")
        val_sizes = val_sizes.item()
        val_labels = np.load(PICKLE_SAVE_FOLDER + "val_labels.npy")
        val_domains = np.load(PICKLE_SAVE_FOLDER + "val_domains.npy")
 
        print("Loading test vectors...")
        f.write("Loading test vectors...\n")
        test_labels = np.load(PICKLE_SAVE_FOLDER + "test_labels.npy")
        test_domains = np.load(PICKLE_SAVE_FOLDER + "test_domains.npy")

        if USE_TF_VECTORS or USE_RELATIONAL_FEATURE_VECTORS or USE_AVG_EMBEDDINGS or ADD_FEATURES_TO_LABEL_PRED:
            print("Loading TF vectors...")
            f.write("Loading TF vectors...\n")
            
            train_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "train_tf_vectors.npy")
            val_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "val_tf_vectors.npy")
            test_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "test_tf_vectors.npy")
            
            SIZE_TRAIN = len(train_tf_vectors)
            SIZE_VAL = len(val_tf_vectors)
            SIZE_TEST = len(test_tf_vectors)

        if USE_RELATIONAL_FEATURE_VECTORS:
            print("Loading relation vectors...")
            f.write("Loading relation vectors...\n")
            
            train_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "train_relation_vectors.npy")
            val_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "val_relation_vectors.npy")
            test_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "test_relation_vectors.npy")
            
            SIZE_TRAIN = len(train_relation_vectors)
            SIZE_VAL = len(val_relation_vectors)
            SIZE_TEST = len(test_relation_vectors)

        if USE_AVG_EMBEDDINGS:
            print("Loading avg embedding vectors...")
            f.write("Loading avg embeddings vectors...\n")
            
            train_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "train_avg_embed_vectors.npy")
            val_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "val_avg_embed_vectors.npy")
            test_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy")
            
            SIZE_TRAIN = len(train_avg_embed_vectors)
            SIZE_VAL = len(val_avg_embed_vectors)
            SIZE_TEST = len(test_avg_embed_vectors)

        if USE_CNN_FEATURES:
            print("Loading CNN vectors...")
            f.write("Loading CNN vectors...\n")
            
            x_train_headlines = np.load(PICKLE_SAVE_FOLDER + "x_train_headlines.npy")
            x_train_bodies = np.load(PICKLE_SAVE_FOLDER + "x_train_bodies.npy")
            x_val_headlines = np.load(PICKLE_SAVE_FOLDER + "x_val_headlines.npy")
            x_val_bodies = np.load(PICKLE_SAVE_FOLDER + "x_val_bodies.npy")
            x_test_headlines = np.load(PICKLE_SAVE_FOLDER + "x_test_headlines.npy")
            x_test_bodies = np.load(PICKLE_SAVE_FOLDER + "x_test_bodies.npy")
            
            SIZE_TRAIN = len(x_train_headlines)
            SIZE_VAL = len(x_val_headlines)
            SIZE_TEST = len(x_test_headlines)

            embedding_matrix = np.load(PICKLE_SAVE_FOLDER + "embedding_matrix.npy")
            word_index = np.load(PICKLE_SAVE_FOLDER + "word_index.npy")
            word_index = word_index.item()

        print("SIZE_TRAIN = ", SIZE_TRAIN)
        f.write("SIZE_TRAIN = " + str(SIZE_TRAIN) + "\n")

        print("SIZE_VAL = ", SIZE_VAL)
        f.write("SIZE_VAL = " + str(SIZE_VAL) + "\n")
 
        print("SIZE_TEST = ", SIZE_TEST)
        f.write("SIZE_TEST = " + str(SIZE_TEST) + "\n")

        ################
        # DEFINE MODEL #
        ################

        best_loss = float('Inf')
        
        for model_num in range(NUM_MODELS_TO_TRAIN):
            print("Training model " + str(model_num))
            f.write("Training model " + str(model_num) + "\n")
            tf.reset_default_graph()
            
            print("Defining Model...")
            f.write("Defining Model...\n")
            if USE_TF_VECTORS:
                FEATURE_VECTOR_SIZE = len(train_tf_vectors[0])
            elif USE_RELATIONAL_FEATURE_VECTORS:
                FEATURE_VECTOR_SIZE = len(train_relation_vectors[0])
            elif USE_AVG_EMBEDDINGS:
                FEATURE_VECTOR_SIZE = len(train_avg_embed_vectors[0])
            elif USE_CNN_FEATURES:
                FEATURE_VECTOR_SIZE = NUM_FILTERS * len(FILTER_SIZES) * 2
        
            stances_pl = tf.placeholder(tf.int64, [None], name="stances_pl")
            keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob_pl")
            domains_pl = tf.placeholder(tf.int64, [None], name="domains_pl")
            gr_pl = tf.placeholder(tf.float32, [], name="gr_pl")
            lr_pl = tf.placeholder(tf.float32, [], name="lr_pl")
            
            if USE_TF_VECTORS or USE_RELATIONAL_FEATURE_VECTORS or USE_AVG_EMBEDDINGS:
                features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE], name="features_pl")

            if ADD_FEATURES_TO_LABEL_PRED:
                p_features_pl = tf.placeholder(tf.float32, [None, len(train_tf_vectors[0])], name="p_features_pl")

            if USE_AVG_EMBEDDINGS:
                avg_embed_vector_pl = tf.placeholder(tf.float32, [None, EMBEDDING_DIM * 2])

            if USE_CNN_FEATURES:
                embedding_matrix_pl = tf.placeholder(tf.float32, [len(word_index) + 1, EMBEDDING_DIM])
                W = tf.Variable(tf.constant(0.0, shape=[len(word_index) + 1, EMBEDDING_DIM]), trainable=False)
                embedding_init = W.assign(embedding_matrix_pl)
                headline_words_pl = tf.placeholder(tf.int64, [None, len(x_train_headlines[0])])
                body_words_pl = tf.placeholder(tf.int64, [None, len(x_train_bodies[0])])
            
            if USE_CNN_FEATURES:
                batch_size = tf.shape(headline_words_pl)[0]
            else:
                batch_size = tf.shape(features_pl)[0]
            
            ### Feature Extraction ###
            
            # TF and TFIDF features fully connected hidden layer of HIDDEN_SIZE
            if USE_CNN_FEATURES:
                pooled_outputs = []

                headline_embeddings = tf.nn.embedding_lookup(embedding_init, headline_words_pl)
                body_embeddings = tf.nn.embedding_lookup(embedding_init, body_words_pl)

                for filter_size in FILTER_SIZES:
                    b_head = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
                    conv_head = tf.layers.conv1d(headline_embeddings, NUM_FILTERS, filter_size)
                    relu_head = tf.nn.relu(tf.nn.bias_add(conv_head, b_head))
                    pool_head = tf.layers.max_pooling1d(relu_head, CNN_HEADLINE_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_head)

                for filter_size in FILTER_SIZES:
                    b_body = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
                    conv_body = tf.layers.conv1d(body_embeddings, NUM_FILTERS, filter_size)
                    relu_body = tf.nn.relu(tf.nn.bias_add(conv_body, b_head))
                    pool_body = tf.layers.max_pooling1d(relu_body, CNN_BODY_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_body)

                cnn_out_vector = tf.reshape(tf.concat(pooled_outputs, 2), [-1, NUM_FILTERS * len(FILTER_SIZES) * 2])
                cnn_out_vector = tf.nn.dropout(cnn_out_vector, keep_prob_pl)
                hidden_layer = tf.layers.dense(cnn_out_vector, HIDDEN_SIZE)

            else:
                hidden_layer = tf.nn.dropout(tf.nn.relu(tf.layers.dense(features_pl, HIDDEN_SIZE)), keep_prob=keep_prob_pl)

            ### Label Prediction ###

            # Fully connected hidden layer with size based on LABEL_HIDDEN_SIZE with original features concated
            if ADD_FEATURES_TO_LABEL_PRED:
                hidden_layer_p = tf.concat([p_features_pl, tf.identity(hidden_layer)], axis=1)
            else:
                hidden_layer_p = hidden_layer
            
            if LABEL_HIDDEN_SIZE is not None:
                hidden_layer_p = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer_p, LABEL_HIDDEN_SIZE)), keep_prob=keep_prob_pl)

            logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer_p, TARGET_SIZE), keep_prob=keep_prob_pl)
            logits = tf.reshape(logits_flat, [batch_size, TARGET_SIZE])

            # Label loss and prediction
            p_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl))
            softmaxed_logits = tf.nn.softmax(logits)
            predict = tf.argmax(softmaxed_logits, axis=1)
            
            ### Domain Prediction ###

            if USE_DOMAINS:
                # Gradient reversal layer
                hidden_layer_d = flip_gradient(hidden_layer, gr_pl)

                # Hidden layer size based on DOMAIN_HIDDEN_SIZE
                if DOMAIN_HIDDEN_SIZE is None:
                    domain_layer = hidden_layer_d
                else:
                    domain_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer_d, DOMAIN_HIDDEN_SIZE)), keep_prob=keep_prob_pl)
                
                d_logits_flat = tf.nn.dropout(tf.contrib.layers.linear(domain_layer, DOMAIN_TARGET_SIZE), keep_prob=keep_prob_pl)
                d_logits = tf.reshape(d_logits_flat, [batch_size, DOMAIN_TARGET_SIZE])

                # Domain loss and prediction
                d_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_logits, labels=domains_pl))
                softmaxed_d_logits = tf.nn.softmax(d_logits)
                d_predict = tf.argmax(softmaxed_d_logits, axis=1)
            
            else:
                d_loss = tf.constant(0.0)
                d_predict = tf.constant(0.0)

            ### Regularization ###
            # L2 loss
            tf_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * L2_ALPHA
           
            # Define optimiser
            opt_func = tf.train.AdamOptimizer(lr_pl)
            grads, _ = tf.clip_by_global_norm(tf.gradients(RATIO_LOSS * p_loss + (1 - RATIO_LOSS) * d_loss + l2_loss, tf_vars), CLIP_RATIO)
            opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
 
            # Intialize saver to save model
            saver = tf.train.Saver()
 
            ### Training Model ###
            
            print("Training Model...")
            f.write("Training Model...\n")
            
            with tf.Session() as sess:
                # Load Pretrained Model if applicable
                if PRETRAINED_MODEL_PATH is not None:
                    print("Loading Saved Model...")
                    f.write("Loading Saved Model...\n")
                    saver.restore(sess, PRETRAINED_MODEL_PATH)
                    
                else:
                    sess.run(tf.global_variables_initializer())
                
                for epoch in range(EPOCH_START, EPOCH_START + EPOCHS):
                    print("\n  EPOCH", epoch)
                    f.write("\n  EPOCH " + str(epoch))
                    
                    # Adaption Parameter and Learning Rate
                    p = float(epoch)/TOTAL_EPOCHS
                    gr = 2. / (1. + np.exp(-10. * p)) - 1
                    lr = LR_FACTOR / (1. + 10 * p)**0.75
                
                    train_loss, train_p_loss, train_d_loss, train_l2_loss = 0, 0, 0, 0
                    train_l_pred, train_d_pred = [], []
                    
                    # Organize and randomize order of training data from each dataset
                    index = 0
                    fnc_indices, snli_indices, fever_indices = [], [], []

                    if USE_FNC_DATA:
                        fnc_indices = list(range(index, index + train_sizes['fnc']))
                        index += train_sizes['fnc']
                        rand1.shuffle(fnc_indices)
                        for i in fnc_indices:
                            assert train_domains[i] == 0
                    
                    if USE_FEVER_DATA:
                        fever_indices = list(range(index, index + train_sizes['fever']))
                        index += train_sizes['fever']
                        rand1.shuffle(fever_indices)
                        for i in fever_indices:
                            assert train_domains[i] == 2
                    
                    if USE_SNLI_DATA:
                        snli_indices = list(range(index, index + train_sizes['snli']))
                        index += train_sizes['snli']
                        rand1.shuffle(snli_indices)
                        for i in snli_indices:
                            assert train_domains[i] == 1
                       
                    # Use equal numbers of FNC and other data per epoch
                    if EXTRA_SAMPLES_PER_EPOCH is not None and USE_FNC_DATA:
                        
                        # Use equal numbers of agree/disagree labels per epoch
                        if BALANCE_LABELS:
                            fnc_agree_indices = [i for i in fnc_indices if train_labels[i] == 0]
                            fnc_disagree_indices = [i for i in fnc_indices if train_labels[i] == 1]
                            
                            snli_agree_indices = [i for i in snli_indices if train_labels[i] == 0]
                            snli_disagree_indices = [i for i in snli_indices if train_labels[i] == 1]

                            fever_agree_indices = [i for i in fever_indices if train_labels[i] == 0]
                            fever_disagree_indices = [i for i in fever_indices if train_labels[i] == 1]

                            LABEL_SIZE = float('inf')
                            if USE_FNC_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(fnc_agree_indices), len(fnc_disagree_indices))
                            if USE_SNLI_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(snli_agree_indices), len(snli_disagree_indices))
                            if USE_FEVER_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(fever_agree_indices), len(fever_disagree_indices))

                            train_indices = fnc_agree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            fnc_disagree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            snli_agree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            snli_disagree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            fever_agree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            fever_disagree_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH]
                        
                        # Don't balance labels but maintain same amount of data from each dataset
                        else:
                            LABEL_SIZE = float('inf')
                            if USE_FNC_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(fnc_indices))
                            if USE_FEVER_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(fever_indices))
                            if USE_SNLI_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(snli_indices))
                            train_indices = fnc_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            snli_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH] + \
                                            fever_indices[:LABEL_SIZE * EXTRA_SAMPLES_PER_EPOCH]
                    
                    # Use all training data each epoch
                    else:
                        train_indices = fnc_indices + snli_indices + fnc_indices

                    # Randomize order of training data
                    rand2.shuffle(train_indices)
                        
                    # Training epoch loop
                    for i in range(len(train_indices) // BATCH_SIZE + 1):

                        # Get training batches
                        batch_indices = train_indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                        if len(batch_indices) == 0:
                            break
                        batch_stances = [train_labels[i] for i in batch_indices]
                        batch_domains = [train_domains[i] for i in batch_indices]
                       
                        batch_feed_dict = {stances_pl: batch_stances,
                                           keep_prob_pl: TRAIN_KEEP_PROB,
                                           lr_pl: lr}
                        
                        if USE_DOMAINS:
                            batch_feed_dict[domains_pl] = batch_domains
                            batch_feed_dict[gr_pl] = gr

                        if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
                            batch_features = [train_tf_vectors[i] for i in batch_indices]
                            if USE_TF_VECTORS:
                                batch_feed_dict[features_pl] = batch_features
                            if ADD_FEATURES_TO_LABEL_PRED:
                                batch_feed_dict[p_features_pl] = batch_features
                        
                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relation_vectors = [train_relation_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_relation_vectors
                        
                        if USE_AVG_EMBEDDINGS:
                            batch_avg_embed_vectors = [train_avg_embed_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_avg_embed_vectors

                        if USE_CNN_FEATURES:
                            batch_x_headlines = [x_train_headlines[i] for i in batch_indices]
                            batch_x_bodies = [x_train_bodies[i] for i in batch_indices]
                            batch_feed_dict[headline_words_pl] = batch_x_headlines
                            batch_feed_dict[body_words_pl] = batch_x_bodies
                            batch_feed_dict[embedding_matrix_pl] = embedding_matrix
                        
                        #Loss and predictions for each batch
                        _, lpred, dpred, ploss, dloss, l2loss = \
                                sess.run([opt_op, predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=dict(batch_feed_dict))
                        
                        # Total loss for current epoch
                        train_p_loss += ploss
                        train_d_loss += dloss
                        train_l2_loss += l2loss
                        train_loss += ploss + dloss + l2loss
                        train_l_pred.extend(lpred)
                        
                        if USE_DOMAINS:
                            train_d_pred.extend(dpred)
                    
                    # Record loss and accuracy information for train
                    actual_train_labels = [train_labels[i] for i in train_indices]
                    actual_train_domains = [train_domains[i] for i in train_indices]
                    print_model_results(f, "Train", train_l_pred, actual_train_labels, train_d_pred, actual_train_domains,
                                        train_p_loss, train_d_loss, train_l2_loss, USE_DOMAINS)

                    # Record loss and accuracy for val
                    if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
                        val_indices = list(range(SIZE_VAL))
                        val_loss, val_p_loss, val_d_loss, val_l2_loss = 0, 0, 0, 0
                        val_l_pred, val_d_pred = [], []
   
                        for i in range(int(SIZE_VAL) // BATCH_SIZE + 1):
                            batch_indices = val_indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                            if len(batch_indices) == 0:
                                break
                            batch_stances = [val_labels[i] for i in batch_indices]
                            batch_domains = [val_domains[i] for i in batch_indices]
                            
                            batch_feed_dict = {stances_pl: batch_stances,
                                               keep_prob_pl: 1.0}

                            if USE_DOMAINS:
                                batch_feed_dict[gr_pl] = 1.0
                                batch_feed_dict[domains_pl] = batch_domains

                            if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
                                batch_features = [val_tf_vectors[i] for i in batch_indices]
                                if USE_TF_VECTORS:
                                    batch_feed_dict[features_pl] = batch_features
                                if ADD_FEATURES_TO_LABEL_PRED:
                                    batch_feed_dict[p_features_pl] = batch_features
                            
                            if USE_RELATIONAL_FEATURE_VECTORS:
                                batch_relation_vectors = [val_relation_vectors[i] for i in batch_indices]
                                batch_feed_dict[features_pl] = batch_relation_vectors
                            
                            if USE_AVG_EMBEDDINGS:
                                batch_avg_embed_vectors = [val_avg_embed_vectors[i] for i in batch_indices]
                                batch_feed_dict[features_pl] = batch_avg_embed_vectors

                            if USE_CNN_FEATURES:
                                batch_x_headlines = [x_val_headlines[i] for i in batch_indices]
                                batch_x_bodies = [x_val_bodies[i] for i in batch_indices]
                                batch_feed_dict[headline_words_pl] = batch_x_headlines
                                batch_feed_dict[body_words_pl] = batch_x_bodies
                                batch_feed_dict[embedding_matrix_pl] = embedding_matrix
                            
                            # Record loss and accuracy information for test
                            lpred, dpred, ploss, dloss, l2loss = \
                                    sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)
                            
                            # Total loss for current epoch
                            val_p_loss += ploss
                            val_d_loss += dloss
                            val_l2_loss += l2loss
                            val_loss += ploss + dloss + l2loss
                            val_l_pred.extend(lpred)
                            if USE_DOMAINS:
                                val_d_pred.extend(dpred)
                        
                        print_model_results(f, "Val", val_l_pred, val_labels, val_d_pred, val_domains,
                                            val_p_loss, val_d_loss, val_l2_loss, USE_DOMAINS)

                        # Save best test label loss model
                        if val_p_loss < best_loss:
                            best_loss = val_p_loss
                            saver.save(sess, SAVE_MODEL_PATH)
                            print("\n    New Best Val Loss")
                            f.write("\n    New Best Val Loss\n")

                    test_indices = list(range(SIZE_TEST))
                    test_loss, test_p_loss, test_d_loss, test_l2_loss = 0, 0, 0, 0
                    test_l_pred, test_d_pred = [], []

                    for i in range(SIZE_TEST // BATCH_SIZE + 1):

                        batch_indices = test_indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                        if len(batch_indices) == 0:
                            break
                        batch_stances = [test_labels[i] for i in batch_indices]
                        batch_domains = [test_domains[i] for i in batch_indices]
                        
                        batch_feed_dict = {stances_pl: batch_stances,
                                           keep_prob_pl: 1.0}

                        if USE_DOMAINS:
                            batch_feed_dict[gr_pl] = 1.0
                            batch_feed_dict[domains_pl] = batch_domains

                        if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
                            batch_features = [test_tf_vectors[i] for i in batch_indices]
                            if USE_TF_VECTORS:
                                batch_feed_dict[features_pl] = batch_features
                            if ADD_FEATURES_TO_LABEL_PRED:
                                batch_feed_dict[p_features_pl] = batch_features
                        
                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relation_vectors = [test_relation_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_relation_vectors
                        
                        if USE_AVG_EMBEDDINGS:
                            batch_avg_embed_vectors = [test_avg_embed_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_avg_embed_vectors

                        if USE_CNN_FEATURES:
                            batch_x_headlines = [x_test_headlines[i] for i in batch_indices]
                            batch_x_bodies = [x_test_bodies[i] for i in batch_indices]
                            batch_feed_dict[headline_words_pl] = batch_x_headlines
                            batch_feed_dict[body_words_pl] = batch_x_bodies
                            batch_feed_dict[embedding_matrix_pl] = embedding_matrix
                        
                        # Record loss and accuracy information for test
                        lpred, dpred, ploss, dloss, l2loss = \
                                sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)
                        
                        # Total loss for current epoch
                        test_p_loss += ploss
                        test_d_loss += dloss
                        test_l2_loss += l2loss
                        test_loss += ploss + dloss + l2loss
                        test_l_pred.extend(lpred)
                        if USE_DOMAINS:
                            test_d_pred.extend(dpred)
                    
                    # Print and Write test results
                    print_model_results(f, "Test", test_l_pred, test_labels, test_d_pred, test_domains,
                                    test_p_loss, test_d_loss, test_l2_loss, USE_DOMAINS)
                    
                #saver.restore(sess, SAVE_MODEL_PATH)
                #test_l_pred = sess.run([predict], feed_dict = test_feed_dict)
                #save_predictions(test_l_pred, test_labels, PREDICTION_FILE)

if __name__ == "__main__":
    process_data()
    #train_model()


