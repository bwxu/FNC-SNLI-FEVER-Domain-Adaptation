
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
from util import get_relational_feature_vectors, remove_stop_words, get_average_embeddings, print_model_results

# Data Processing Params
PICKLE_SAVE_FOLDER = "pickle_data/2_label_fever_a_d/"
PICKLE_LOG_FILE = PICKLE_SAVE_FOLDER + "log.txt"

# Saving Parameters
MODEL_NAME = "fever_tf_a_d"
SAVE_FOLDER = "models/apr_19/" + MODEL_NAME + "/"
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
USE_FNC_DATA = True
USE_SNLI_DATA = False
USE_FEVER_DATA = True
USE_SNLI_NEUTRAL = False
SNLI_TOTAL_SAMPLES_LIMIT = None
ONLY_VECT_FNC = True

assert not USE_SNLI_DATA or not USE_FEVER_DATA

USE_DOMAINS = False

ADD_FEATURES_TO_LABEL_PRED = False

USE_TF_VECTORS = True
USE_RELATIONAL_FEATURE_VECTORS = False
USE_AVG_EMBEDDINGS = False
USE_CNN_FEATURES = False
input_vectors = [USE_TF_VECTORS, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_CNN_FEATURES]  
#assert len([x for x in input_vectors if x == True]) <= 1 

EPOCHS = 30
TOTAL_EPOCHS = 30
EPOCH_START = 0
VALIDATION_SET_SIZE = 0.2
NUM_MODELS_TO_TRAIN = 5

if not USE_FNC_DATA and not USE_SNLI_DATA and not USE_FEVER_DATA:
    raise Exception("Must choose data to use")

EXTRA_SAMPLES_PER_EPOCH = None
if USE_UNRELATED_LABEL:
    FNC_SIZE = 49972
    if USE_FNC_DATA:
        EXTRA_SAMPLES_PER_EPOCH = 49972
else:
    FNC_SIZE = 13427
    if USE_FNC_DATA:
        EXTRA_SAMPLES_PER_EPOCH = 13427
    if not USE_DISCUSS_LABEL:
        FNC_SIZE = 4518
        if USE_FNC_DATA:
            EXTRA_SAMPLES_PER_EPOCH = 4518

RATIO_LOSS = 0.5

CHECKS_ENABLED = False
if CHECKS_ENABLED:
    if not USE_UNRELATED_LABEL:
        assert "3_label" in PICKLE_SAVE_FOLDER
    if USE_SNLI_NEUTRAL:
        assert "use_neutral" in PICKLE_SAVE_FOLDER
    if USE_SNLI_NEUTRAL:
        assert "_d_" in MODEL_NAME
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
DOMAIN_TARGET_SIZE = 2
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
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL\n")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        ### Extract Data ###

        print("Getting Data...")
        f.write("Getting Data...\n")
        
        # for now only supports when using FNC data
        assert USE_FNC_DATA == True

        fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_body_ids_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
        fnc_domain = [0 for _ in range(len(fnc_headlines_train))]
        fnc_headlines_test, fnc_bodies_test, fnc_labels_test, fnc_body_ids_test = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)
        
        if USE_SNLI_DATA:
            snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=SNLI_TOTAL_SAMPLES_LIMIT, use_neutral=USE_SNLI_NEUTRAL)
            snli_domain = [1 for _ in range(len(snli_s1_train))]

        if USE_FEVER_DATA:
            fever_headlines, fever_bodies, fever_labels, fever_claim_set = get_fever_data(FEVER_TRAIN, FEVER_WIKI)
            fever_domain = [1 for _ in range(len(fever_headlines))]

        extra_headlines = []
        extra_bodies = []
        extra_labels = []
        extra_domains = []

        if USE_SNLI_DATA:
            extra_headlines += snli_s1_train
            extra_bodies += snli_s2_train
            extra_labels += snli_labels_train
            extra_domains += snli_domain

        elif USE_FEVER_DATA:
            extra_headlines += fever_headlines
            extra_bodies += fever_bodies
            extra_labels += fever_labels
            extra_domains += fever_domain

        train_headlines = fnc_headlines_train + extra_headlines
        train_bodies = fnc_bodies_train + extra_bodies
        train_labels = fnc_labels_train + extra_labels
        train_domains = fnc_domain + extra_domains

        if not USE_UNRELATED_LABEL:
            unrelated_indices = [i for i, x in enumerate(train_labels) if x == 3]
            for i in sorted(unrelated_indices, reverse=True):
                del train_headlines[i]
                del train_bodies[i]
                del train_labels[i]
                del train_domains[i]

        if not USE_DISCUSS_LABEL:
            unrelated_indices = [i for i, x in enumerate(train_labels) if x == 2]
            for i in sorted(unrelated_indices, reverse=True):
                del train_headlines[i]
                del train_bodies[i]
                del train_labels[i]
                del train_domains[i]
        
        print(len(train_labels))

        np.save(PICKLE_SAVE_FOLDER + "train_labels.npy", np.asarray(train_labels))
        del train_labels
        np.save(PICKLE_SAVE_FOLDER + "train_domains.npy", np.asarray(train_domains))
        del train_domains

        test_headlines = fnc_headlines_test
        test_bodies = fnc_bodies_test
        test_labels = fnc_labels_test
        test_domains = [0 for _ in range(len(test_headlines))]
        
        if not USE_UNRELATED_LABEL:
            unrelated_indices = [i for i, x in enumerate(test_labels) if x == 3]
            for i in sorted(unrelated_indices, reverse=True):
                del test_headlines[i]
                del test_bodies[i]
                del test_labels[i]
                del test_domains[i]

        if not USE_DISCUSS_LABEL:
            unrelated_indices = [i for i, x in enumerate(test_labels) if x == 2]
            for i in sorted(unrelated_indices, reverse=True):
                del test_headlines[i]
                del test_bodies[i]
                del test_labels[i]
                del test_domains[i]
 
        print(len(test_labels))
        
        np.save(PICKLE_SAVE_FOLDER + "test_labels.npy", np.asarray(test_labels))
        del test_labels
        np.save(PICKLE_SAVE_FOLDER + "test_domains.npy", np.asarray(test_domains))
        del test_domains

        ### Get TF and TFIDF vectorizers ###

        print("Creating Vectorizers...")
        f.write("Creating Vectorizers...\n")

        vec_train_data = train_headlines + train_bodies
        vec_test_data = fnc_headlines_test + fnc_bodies_test

        # Only train vectorizers with FNC data
        if ONLY_VECT_FNC:
            vec_train_data = fnc_headlines_train + fnc_bodies_train
            vec_test_data = fnc_headlines_test + fnc_bodies_test
        
        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(vec_train_data, vec_test_data, MAX_FEATURES)
        del vec_train_data
        del vec_test_data

        ### Get Feature Vectors ###
        if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
            print("Getting Feature Vectors...")
            f.write("Getting Feature Vectors...\n")
            
            print("  FNC/SNLI train...")
            f.write("  FNC/SNLI train...\n")
            train_tf_vectors = get_feature_vectors(train_headlines, train_bodies,
                                                   bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(PICKLE_SAVE_FOLDER + "train_tf_vectors.npy", train_tf_vectors)
            f.write("    Number of Feature Vectors: " + str(len(train_tf_vectors)) + "\n")

            print("  FNC test...")
            f.write("  FNC test...\n")
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
            tokenizer.fit_on_texts(train_headlines + test_bodies)
            train_headline_seq = tokenizer.texts_to_sequences(train_headlines)
            train_body_seq = tokenizer.texts_to_sequences(train_bodies)
            test_headline_seq = tokenizer.texts_to_sequences(test_headlines)
            test_body_seq = tokenizer.texts_to_sequences(test_bodies)
            word_index = tokenizer.word_index
            np.save(PICKLE_SAVE_FOLDER + "word_index.npy", word_index)

            print("  Padding Sequences...")
            f.write("  Padding Sequences...\n")
            x_train_headlines = pad_sequences(train_headline_seq, maxlen=CNN_HEADLINE_LENGTH)
            x_train_bodies = pad_sequences(train_body_seq, maxlen=CNN_BODY_LENGTH)
            x_test_headlines = pad_sequences(test_headline_seq, maxlen=CNN_HEADLINE_LENGTH)
            x_test_bodies = pad_sequences(test_body_seq, maxlen=CNN_BODY_LENGTH)

            np.save(PICKLE_SAVE_FOLDER + "x_train_headlines.npy", x_train_headlines)
            np.save(PICKLE_SAVE_FOLDER + "x_train_bodies.npy", x_train_bodies)
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
                USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL, USE_TF_VECTORS, USE_CNN_FEATURES]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, \
               USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL, USE_TF_VECTORS, USE_CNN_FEATURES")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, \
                USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL, USE_TF_VECTORS, USE_CNN_FEATURES\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        print("Loading train vectors...")
        f.write("Loading train vectors...\n")

        # Take last VALDIATION_SET_SIZE PROPORTION of train set as validation set
        train_labels = np.load(PICKLE_SAVE_FOLDER + "train_labels.npy")
        train_domains = np.load(PICKLE_SAVE_FOLDER + "train_domains.npy")
        
        print("Loading test vectors...")
        f.write("Loading test vectors...\n")
        test_labels = np.load(PICKLE_SAVE_FOLDER + "test_labels.npy")
        test_domains = np.load(PICKLE_SAVE_FOLDER + "test_domains.npy")

        if USE_TF_VECTORS or USE_RELATIONAL_FEATURE_VECTORS or USE_AVG_EMBEDDINGS or ADD_FEATURES_TO_LABEL_PRED:
            print("Loading TF vectors...")
            f.write("Loading TF vectors...\n")
            train_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "train_tf_vectors.npy")
            test_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "test_tf_vectors.npy")
            SIZE_TRAIN = len(train_tf_vectors)
            SIZE_TEST = len(test_tf_vectors)

        if USE_RELATIONAL_FEATURE_VECTORS:
            print("Loading relation vectors...")
            f.write("Loading relation vectors...\n")
            train_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "train_relation_vectors.npy")
            test_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "test_relation_vectors.npy")

        if USE_AVG_EMBEDDINGS:
            print("Loading avg embedding vectors...")
            f.write("Loading avg embeddings vectors...\n")
            train_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "train_avg_embed_vectors.npy")
            test_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy")
            SIZE_TRAIN = len(train_avg_embed_vectors)
            SIZE_TEST = len(test_avg_embed_vectors)

        if USE_CNN_FEATURES:
            print("Loading CNN vectors...")
            f.write("Loading CNN vectors...\n")
            x_train_headlines = np.load(PICKLE_SAVE_FOLDER + "x_train_headlines.npy")
            x_train_bodies = np.load(PICKLE_SAVE_FOLDER + "x_train_bodies.npy")
            x_test_headlines = np.load(PICKLE_SAVE_FOLDER + "x_test_headlines.npy")
            x_test_bodies = np.load(PICKLE_SAVE_FOLDER + "x_test_bodies.npy")
            SIZE_TRAIN = len(x_train_headlines)
            SIZE_TEST = len(x_test_headlines)

            embedding_matrix = np.load(PICKLE_SAVE_FOLDER + "embedding_matrix.npy")
            word_index = np.load(PICKLE_SAVE_FOLDER + "word_index.npy")
            word_index = word_index.item()


        if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
            print("Getting validation vectors...")
            f.write("Getting validation vectors...\n")

            if (not USE_SNLI_DATA and not USE_FEVER_DATA) or not USE_FNC_DATA:
                VAL_SIZE_PER_SET = int(SIZE_TRAIN * VALIDATION_SET_SIZE)
                val_labels = train_labels[-VAL_SIZE_PER_SET:]
                val_domains = train_domains[-VAL_SIZE_PER_SET:]
        
                train_labels = train_labels[:-VAL_SIZE_PER_SET]
                train_domains = train_domains[:-VAL_SIZE_PER_SET]

                if USE_TF_VECTORS or USE_RELATIONAL_FEATURE_VECTORS or USE_AVG_EMBEDDINGS or ADD_FEATURES_TO_LABEL_PRED:
                    val_tf_vectors = train_tf_vectors[-VAL_SIZE_PER_SET:, :]
                    train_tf_vectors = train_tf_vectors[:-VAL_SIZE_PER_SET, :]

                if USE_RELATIONAL_FEATURE_VECTORS:
                    val_relation_vectors = train_relation_vectors[-VAL_SIZE_PER_SET:, :]
                    train_relation_vectors = train_relation_vectors[:-VAL_SIZE_PER_SET, :]

                if USE_AVG_EMBEDDINGS:
                    val_avg_embed_vectors = train_avg_embed_vectors[-VAL_SIZE_PER_SET:, :]
                    train_avg_embed_vectors = train_avg_embed_vectors[:-VAL_SIZE_PER_SET, :]

                if USE_CNN_FEATURES:
                    x_val_headlines = x_train_headlines[-VAL_SIZE_PER_SET:, :]
                    x_train_headlines = x_train_headlines[:-VAL_SIZE_PER_SET, :]
                    
                    x_val_bodies = x_train_bodies[-VAL_SIZE_PER_SET:, :]
                    x_train_bodies = x_train_bodies[:-VAL_SIZE_PER_SET, :]

            # if both datasets are present take VAL_SIZE_PER_SET from each for the validation set
            else:
                VAL_SIZE_PER_SET = int(FNC_SIZE * VALIDATION_SET_SIZE)
                val_labels = np.concatenate([train_labels[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE], \
                                             train_labels[-VAL_SIZE_PER_SET:]])
                val_domains = np.concatenate([train_domains[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE], \
                                              train_domains[-VAL_SIZE_PER_SET:]])
                
                train_labels = np.concatenate([train_labels[:FNC_SIZE - VAL_SIZE_PER_SET], \
                                               train_labels[FNC_SIZE:-VAL_SIZE_PER_SET]])
                train_domains = np.concatenate([train_domains[:FNC_SIZE - VAL_SIZE_PER_SET], \
                                                train_domains[FNC_SIZE:-VAL_SIZE_PER_SET]])
                
                if USE_TF_VECTORS or ADD_FEATURES_TO_LABEL_PRED:
                    val_tf_vectors = np.concatenate([train_tf_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :], \
                                                     train_tf_vectors[-VAL_SIZE_PER_SET:, :]])
                    train_tf_vectors = np.concatenate([train_tf_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :], \
                                                       train_tf_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]])

                if USE_RELATIONAL_FEATURE_VECTORS:
                    val_relation_vectors = np.concatenate([train_relation_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :], \
                                                           train_relation_vectors[-VAL_SIZE_PER_SET:, :]])
                    train_relation_vectors = np.concatenate([train_relation_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :], \
                                                             train_relation_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]])

                if USE_AVG_EMBEDDINGS:
                    val_relation_vectors = np.concatenate([train_avg_embed_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :], \
                                                           train_avg_embed_vectors[-VAL_SIZE_PER_SET:, :]])
                    train_relation_vectors = np.concatenate([train_avg_embed_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :], \
                                                             train_avg_embed_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]])

                if USE_CNN_FEATURES:
                    x_val_headlines = np.concatenate([x_train_headlines[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :], \
                                                      x_train_headlines[-VAL_SIZE_PER_SET:, :]])
                    x_train_headlines = np.concatenate([x_train_headlines[:FNC_SIZE - VAL_SIZE_PER_SET, :], \
                                                        x_train_headlines[FNC_SIZE:-VAL_SIZE_PER_SET, :]])

                    x_val_bodies = np.concatenate([x_train_bodies[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :], \
                                                   x_train_bodies[-VAL_SIZE_PER_SET:, :]])
                    x_train_bodies = np.concatenate([x_train_bodies[:FNC_SIZE - VAL_SIZE_PER_SET, :], \
                                                     x_train_bodies[FNC_SIZE:-VAL_SIZE_PER_SET, :]])

        print("SIZE_TRAIN = ", SIZE_TRAIN)
        f.write("SIZE_TRAIN = " + str(SIZE_TRAIN) + "\n")
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
                    
                    # Record loss and accuracy information for test
                    #test_l_pred, test_d_pred, test_p_loss, test_d_loss, test_l2_loss = \
                    #        sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=test_feed_dict)
                    
                    # Print and Write test results
                    #print_model_results(f, "Test Original", test_l_pred, test_labels, test_d_pred, test_domains,
                    #                test_p_loss, test_d_loss, test_l2_loss, USE_DOMAINS)
 

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
                    
                    # Randomize order of FNC and SNLI data and randomly choose EXTRA_SAMPLES_PER_EPOCH SNLI data
                    # If both FNC and SNLI data used ensure that the correct amount of SNLI samples
                    # are used each epoch in addition to all of the FNC data
                    if USE_FNC_DATA and USE_SNLI_DATA or USE_FNC_DATA and USE_FEVER_DATA:
                        if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
                            FNC_TRAIN_SIZE = FNC_SIZE - int(FNC_SIZE * VALIDATION_SET_SIZE)
                        else:
                            FNC_TRAIN_SIZE = FNC_SIZE
                        fnc_indices = list(range(FNC_TRAIN_SIZE))
                        snli_indices = list(range(FNC_SIZE, SIZE_TRAIN - int(SIZE_TRAIN * VALIDATION_SET_SIZE)))
                        rand1.shuffle(snli_indices)

                        if EXTRA_SAMPLES_PER_EPOCH is not None:
                            train_indices = fnc_indices + snli_indices[:EXTRA_SAMPLES_PER_EPOCH]
                        else:
                            train_indices = fnc_indices + snli_indices
                        rand2.shuffle(train_indices)

                    else:
                        train_indices = list(range(SIZE_TRAIN - int(SIZE_TRAIN * VALIDATION_SET_SIZE)))
                        rand1.shuffle(train_indices)
                        
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
                    print_model_results(f, "Train", train_l_pred, train_labels, train_d_pred, train_domains,
                                        train_p_loss, train_d_loss, train_l2_loss, USE_DOMAINS)

                    # Record loss and accuracy for val
                    if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
                        if not USE_SNLI_DATA and not USE_FEVER_DATA:
                            VAL_SIZE = int(SIZE_TRAIN * VALIDATION_SET_SIZE)
                        else:
                            VAL_SIZE = int(FNC_SIZE * VALIDATION_SET_SIZE) * 2

                        val_indices = list(range(VAL_SIZE))
                        val_loss, val_p_loss, val_d_loss, val_l2_loss = 0, 0, 0, 0
                        val_l_pred, val_d_pred = [], []
   
                        for i in range(int(VAL_SIZE) // BATCH_SIZE + 1):
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
    #process_data()
    train_model()


