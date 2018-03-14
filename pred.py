
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
from util import get_fnc_data, get_snli_data, get_vectorizers, get_feature_vectors, save_predictions
from util import get_relational_feature_vectors, remove_stop_words, get_average_embeddings, print_model_results

# Data Processing Params
LOG_FILE = "log.txt"
PICKLE_SAVE_FOLDER = "pickle_data/3_label_only_fnc/"

# Saving Parameters
MODEL_NAME = "snli_pre_fnc_3"
SAVE_FOLDER = "models/" + MODEL_NAME + "/"
PREDICTION_FILE = SAVE_FOLDER + MODEL_NAME + ".csv"
SAVE_MODEL_PATH = SAVE_FOLDER + MODEL_NAME + ".ckpt"
TRAINING_LOG_FILE = SAVE_FOLDER + "training.txt"
if not os.path.exists(SAVE_FOLDER):
    raise Exception("Folder of model name doesn't exist")

PRETRAINED_MODEL_PATH = "models/snli_pretrain/snli_pretrain.ckpt"

# Model options
USE_UNRELATED_LABEL = False
USE_FNC_DATA = True
USE_SNLI_DATA = False
USE_SNLI_NEUTRAL = True
USE_DOMAINS = False
ONLY_VECT_FNC = True
ADD_FEATURES_TO_LABEL_PRED = False
USE_RELATIONAL_FEATURE_VECTORS = False
USE_AVG_EMBEDDINGS = False
SNLI_TOTAL_SAMPLES_LIMIT = None
EPOCHS = 30
TOTAL_EPOCHS = 60
EPOCH_START = 30
VALIDATION_SET_SIZE = 0.2
NUM_MODELS_TO_TRAIN = 20

SNLI_SAMPLES_PER_EPOCH = None
if USE_UNRELATED_LABEL:
    FNC_SIZE = 49972
    if USE_FNC_DATA:
        SNLI_SAMPLES_PER_EPOCH = 49972
else:
    FNC_SIZE = 13427
    if USE_FNC_DATA:
        SNLI_SAMPLES_PER_EPOCH = 13427

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

# Model parameters
rand1 = random.Random()
rand2 = random.Random()
MAX_FEATURES = 5000
TARGET_SIZE = 4
DOMAIN_TARGET_SIZE = 2
HIDDEN_SIZE = 100
DOMAIN_HIDDEN_SIZE = 100
LABEL_HIDDEN_SIZE = 100
TRAIN_KEEP_PROB = 0.6
L2_ALPHA = 0.01
CLIP_RATIO = 5
BATCH_SIZE_TRAIN = 100
LR_FACTOR = 0.01

def process_data():
   with open(LOG_FILE, 'a') as f:
        # Save parameters to log file
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        ### Extract Data ###

        print("Getting Data...")
        f.write("Getting Data...\n")
        
        fnc_headlines_train, fnc_bodies_train, fnc_labels_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
        fnc_domain = [0 for _ in range(len(fnc_headlines_train))]
        fnc_headlines_test, fnc_bodies_test, fnc_labels_test = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)

        snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=SNLI_TOTAL_SAMPLES_LIMIT, use_neutral=USE_SNLI_NEUTRAL)
        snli_domain = [1 for _ in range(len(snli_s1_train))]

        train_headlines = fnc_headlines_train + snli_s1_train
        train_bodies = fnc_bodies_train + snli_s2_train
        train_labels = fnc_labels_train + snli_labels_train
        train_domains = fnc_domain + snli_domain

        if not USE_UNRELATED_LABEL:
            unrelated_indices = [i for i, x in enumerate(train_labels) if x == 3]
            for i in sorted(unrelated_indices, reverse=True):
                del train_headlines[i]
                del train_bodies[i]
                del train_labels[i]
                del train_domains[i]

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

        np.save(PICKLE_SAVE_FOLDER + "test_labels.npy", np.asarray(test_labels))
        del test_labels
        np.save(PICKLE_SAVE_FOLDER + "test_domains.npy", np.asarray(test_domains))
        del test_domains

        ### Get TF and TFIDF vectorizers ###

        print("Creating Vectorizers...")
        f.write("Creating Vectorizers...\n")

        train_data = fnc_headlines_train + snli_s1_train + fnc_bodies_train + snli_s2_train
        test_data = fnc_headlines_test + fnc_bodies_test

        if ONLY_VECT_FNC:
            train_data = fnc_headlines_train + fnc_bodies_train
            test_data = fnc_headlines_test + fnc_bodies_test
        
        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(train_data, test_data, MAX_FEATURES)
        del train_data
        del test_data

        ### Get Feature Vectors ###

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

        if USE_AVG_EMBEDDINGS:
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
            
            del embeddings
            del train_headlines
            del train_bodies
            del test_headlines
            del test_bodies

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


def train_model():
    with open(TRAINING_LOG_FILE, 'w') as f:
        # Loop for training multiple models\
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL]
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL")
        print(vals)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_AVG_EMBEDDINGS, USE_SNLI_NEUTRAL\n")
        f.write(', '.join(str(val) for val in vals) + "\n")

        print("Loading train vectors...")
        f.write("Loading train vectors...\n")

        # Take last VALDIATION_SET_SIZE PROPORTION of train set as validation set
        train_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "train_tf_vectors.npy")
        train_labels = np.load(PICKLE_SAVE_FOLDER + "train_labels.npy")
        train_domains = np.load(PICKLE_SAVE_FOLDER + "train_domains.npy")
        
        # TODO fix this hack
        if "only_fnc" not in PICKLE_SAVE_FOLDER:
            if not USE_FNC_DATA:
                train_tf_vectors = train_tf_vectors[FNC_SIZE:]
                train_labels = train_labels[FNC_SIZE:]
                train_domains = train_domains[FNC_SIZE:]

            if not USE_SNLI_DATA:
                train_tf_vectors = train_tf_vectors[:FNC_SIZE]
                train_labels = train_labels[:FNC_SIZE]
                train_domains = train_domains[:FNC_SIZE]

        print("Loading test vectors...")
        f.write("Loading test vectors...\n")
        test_tf_vectors = np.load(PICKLE_SAVE_FOLDER + "test_tf_vectors.npy")
        test_labels = np.load(PICKLE_SAVE_FOLDER + "test_labels.npy")
        test_domains = np.load(PICKLE_SAVE_FOLDER + "test_domains.npy")

        if USE_RELATIONAL_FEATURE_VECTORS:
            print("Loading relation vectors...")
            f.write("Loading relation vectors...\n")
            train_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "train_relation_vectors.npy")
            test_relation_vectors = np.load(PICKLE_SAVE_FOLDER + "test_relation_vectors.npy")

        if USE_AVG_EMBEDDINGS:
            print("Loading relation vectors...")
            f.write("Loading relation vectors...\n")
            train_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "train_avg_embed_vectors.npy")
            test_avg_embed_vectors = np.load(PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy")


        if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
            print("Getting validation vectors...")
            f.write("Getting validation vectors...")

            if not USE_FNC_DATA or not USE_SNLI_DATA:
                VAL_SIZE_PER_SET = int(len(train_tf_vectors) * VALIDATION_SET_SIZE)
                val_tf_vectors = train_tf_vectors[-VAL_SIZE_PER_SET:, :]
                val_labels = train_labels[-VAL_SIZE_PER_SET:]
                val_domains = train_domains[-VAL_SIZE_PER_SET:]
        
                train_tf_vectors = train_tf_vectors[:-VAL_SIZE_PER_SET, :]
                train_labels = train_labels[:-VAL_SIZE_PER_SET]
                train_domains = train_domains[:-VAL_SIZE_PER_SET]

                if USE_RELATIONAL_FEATURE_VECTORS:
                    val_relation_vectors = train_relation_vectors[-VAL_SIZE_PER_SET:, :]
                    train_relation_vectors = train_relation_vectors[:-VAL_SIZE_PER_SET, :]

                if USE_AVG_EMBEDDINGS:
                    val_avg_embed_vectors = train_avg_embed_vectors[-VAL_SIZE_PER_SET:, :]
                    train_avg_embed_vectors = train_avg_embed_vectors[:-VAL_SIZE_PER_SET, :]

            # if both datasets are present take VAL_SIZE_PER_SET from each for the validation set
            else:
                VAL_SIZE_PER_SET = int(FNC_SIZE * VALIDATION_SET_SIZE)
                val_tf_vectors = train_tf_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :] + \
                        train_tf_vectors[-VAL_SIZE_PER_SET:, :]
                val_labels = train_labels[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE] + \
                        train_labels[-VAL_SIZE_PER_SET:]
                val_domains = train_domains[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE] + \
                        train_domains[-VAL_SIZE_PER_SET:]
                
                train_tf_vectors = train_tf_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :] + \
                        train_tf_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]
                train_labels = train_labels[:FNC_SIZE - VAL_SIZE_PER_SET] + \
                        train_labels[FNC_SIZE:-VAL_SIZE_PER_SET]
                train_domains = train_domains[:FNC_SIZE - VAL_SIZE_PER_SET] + \
                        train_domains[FNC_SIZE:-VAL_SIZE_PER_SET]

                if USE_RELATIONAL_FEATURE_VECTORS:
                    val_relation_vectors = train_relation_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :] + \
                            train_relation_vectors[-VAL_SIZE_PER_SET:, :]
                    train_relation_vectors = train_relation_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :] + \
                            train_relation_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]

                if USE_AVG_EMBEDDINGS:
                    val_relation_vectors = train_avg_embed_vectors[FNC_SIZE - VAL_SIZE_PER_SET:FNC_SIZE, :] + \
                            train_avg_embed_vectors[-VAL_SIZE_PER_SET:, :]
                    train_relation_vectors = train_avg_embed_vectors[:FNC_SIZE - VAL_SIZE_PER_SET, :] + \
                            train_avg_embed_vectors[FNC_SIZE:-VAL_SIZE_PER_SET, :]

        print("SIZE_TRAIN = ", len(train_tf_vectors))
        f.write("SIZE_TRAIN = " + str(len(train_tf_vectors)) + "\n")

        best_loss = float('Inf')
        
        for model_num in range(NUM_MODELS_TO_TRAIN):
            print("Training model " + str(model_num))
            f.write("Training model " + str(model_num) + "\n")
            tf.reset_default_graph()
            
            print("Defining Model...")
            f.write("Defining Model...\n")
            FEATURE_VECTOR_SIZE = len(train_tf_vectors[0])
            if USE_RELATIONAL_FEATURE_VECTORS:
                FEATURE_VECTOR_SIZE = len(train_relation_vectors[0])
            if USE_AVG_EMBEDDINGS:
                FEATURE_VECTOR_SIZE = len(train_avg_embed_vectors[0])
        
            features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE])
            stances_pl = tf.placeholder(tf.int64, [None])
            p_features_pl = tf.placeholder(tf.float32, [None, len(train_tf_vectors[0])])
            keep_prob_pl = tf.placeholder(tf.float32)
            domains_pl = tf.placeholder(tf.int64, [None])
            gr_pl = tf.placeholder(tf.float32, [])
            lr_pl = tf.placeholder(tf.float32, [])

            if USE_AVG_EMBEDDINGS:
                avg_embed_vector_pl = tf.placeholder(tf.float32, [None, EMBEDDING_DIM * 2])
            
            batch_size = tf.shape(features_pl)[0]

            ### Feature Extraction ###
            
            # TF and TFIDF features fully connected hidden layer of HIDDEN_SIZE
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

            # Get validation inputs
            val_feed_dict = {features_pl: val_tf_vectors,
                              p_features_pl: val_tf_vectors,
                              stances_pl: val_labels,
                              keep_prob_pl: 1.0,
                              domains_pl: val_domains,
                              gr_pl: 1.0}

            # Get test inputs
            test_feed_dict = {features_pl: test_tf_vectors,
                              p_features_pl: test_tf_vectors,
                              stances_pl: test_labels,
                              keep_prob_pl: 1.0,
                              domains_pl: test_domains,
                              gr_pl: 1.0}

            if USE_RELATIONAL_FEATURE_VECTORS:
                test_feed_dict = {features_pl: test_relation_vectors,
                                  p_features_pl: test_tf_vectors,
                                  stances_pl: test_labels,
                                  keep_prob_pl: 1.0,
                                  domains_pl: test_domains,
                                  gr_pl: 1.0}

            if USE_AVG_EMBEDDINGS:
                test_feed_dict = {features_pl: test_avg_embed_vectors,
                                  p_features_pl: test_tf_vectors,
                                  stances_pl: test_labels,
                                  keep_prob_pl: 1.0,
                                  domains_pl: test_domains,
                                  gr_pl: 1.0}
 
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
                    
                    # Randomize order of FNC and SNLI data and randomly choose SNLI_SAMPLES_PER_EPOCH SNLI data
                    # If both FNC and SNLI data used ensure that the correct amount of SNLI samples
                    # are used each epoch in addition to all of the FNC data
                    if USE_FNC_DATA and USE_SNLI_DATA:
                        n_train = len(train_tf_vectors)
                        if VALIDATION_SET_SIZE is not None and VALIDATION_SET_SIZE > 0:
                            FNC_TRAIN_SIZE = FNC_SIZE - int(FNC_SIZE * VALIDATION_SET_SIZE)
                        else:
                            FNC_TRAIN_SIZE = FNC_SIZE
                        fnc_indices = list(range(FNC_TRAIN_SIZE))
                        snli_indices = list(range(FNC_TRAIN_SIZE, n_train))
                        rand1.shuffle(snli_indices)

                        if SNLI_SAMPLES_PER_EPOCH is not None:
                            indices = fnc_indices + snli_indices[:SNLI_SAMPLES_PER_EPOCH]
                        else:
                            indices = fnc_indices + snli_indices
                        rand2.shuffle(indices)

                    else:
                        n_train = len(train_tf_vectors)
                        indices = list(range(n_train))
                        rand1.shuffle(indices)
                        
                    # Training epoch loop
                    for i in range(len(indices) // BATCH_SIZE_TRAIN):

                        # Get training batches
                        batch_indices = indices[i * BATCH_SIZE_TRAIN: (i + 1) * BATCH_SIZE_TRAIN]
                        batch_features = [train_tf_vectors[i] for i in batch_indices]
                        batch_stances = [train_labels[i] for i in batch_indices]
                        batch_domains = [train_domains[i] for i in batch_indices]

                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relation_vectors = [train_relation_vectors[i] for i in batch_indices]
                        
                        if USE_AVG_EMBEDDINGS:
                            batch_avg_embed_vectors = [train_avg_embed_vectors[i] for i in batch_indices]

                        batch_feed_dict = {features_pl: batch_features,
                                           p_features_pl: batch_features,
                                           stances_pl: batch_stances,
                                           keep_prob_pl: TRAIN_KEEP_PROB,
                                           domains_pl: batch_domains,
                                           gr_pl: gr,
                                           lr_pl: lr}

                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_feed_dict = {features_pl: batch_relation_vectors,
                                               p_features_pl: batch_features,
                                               stances_pl: batch_stances,
                                               keep_prob_pl: TRAIN_KEEP_PROB,
                                               domains_pl: batch_domains,
                                               gr_pl: gr,
                                               lr_pl: lr}

                        if USE_AVG_EMBEDDINGS:
                            batch_feed_dict = {features_pl: batch_avg_embed_vectors,
                                               p_features_pl: batch_features,
                                               stances_pl: batch_stances,
                                               keep_prob_pl: TRAIN_KEEP_PROB,
                                               domains_pl: batch_domains,
                                               gr_pl: gr,
                                               lr_pl: lr}

                        # Loss and predictions for each batch
                        _, lpred, dpred, ploss, dloss, l2loss = \
                                sess.run([opt_op, predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)
                        
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
                        val_l_pred, val_d_pred, val_p_loss, val_d_loss, val_l2_loss = \
                                sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=val_feed_dict)

                        print_model_results(f, "Val", val_l_pred, val_labels, val_d_pred, val_domains,
                                            val_p_loss, val_d_loss, val_l2_loss, USE_DOMAINS)

                        # Save best test label loss model
                        if val_p_loss < best_loss:
                            best_loss = val_p_loss
                            saver.save(sess, SAVE_MODEL_PATH)
                            print("\n    New Best Val Loss")
                            f.write("\n    New Best Val Loss\n")

                    # Record loss and accuracy information for test
                    test_l_pred, test_d_pred, test_p_loss, test_d_loss, test_l2_loss = \
                            sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=test_feed_dict)
                    
                    # Print and Write test results
                    print_model_results(f, "Test", test_l_pred, test_labels, test_d_pred, test_domains,
                                    test_p_loss, test_d_loss, test_l2_loss, USE_DOMAINS)
                
                save_predictions(test_l_pred, test_labels, PREDICTION_FILE)


if __name__ == "__main__":
    #process_data()
    train_model()

