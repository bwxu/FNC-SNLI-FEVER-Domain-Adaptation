# Copyright 2017 Benjamin Riedel
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
from util import get_fnc_data, get_snli_data, get_vectorizers, get_feature_vectors, save_predictions, get_prediction_accuracies, get_composite_score, get_relational_feature_vectors, remove_stop_words, get_average_embeddings

def main():
    # Mode is either train or load
    MODE = 'train'
    
    # Saving Parameters
    MODEL_NAME = "e_avg_dann_50000"
    SAVE_FOLDER = "models/" + MODEL_NAME + "/"
    PREDICTION_FILE = SAVE_FOLDER + MODEL_NAME + ".csv"
    SAVE_MODEL_PATH = SAVE_FOLDER + MODEL_NAME
    TRAINING_LOG_FILE = SAVE_FOLDER + "training.txt"
    if not os.path.exists(SAVE_FOLDER):
        raise Exception("Folder of model name doesn't exist")

    # Model options
    USE_FNC_DATA = True
    USE_SNLI_NEUTRAL = False
    USE_DOMAINS = True
    ONLY_VECT_FNC = True
    ADD_FEATURES_TO_LABEL_PRED = True
    USE_RELATIONAL_FEATURE_VECTORS = False
    USE_CNN_FEATURES = False
    USE_AVG_EMBEDDINGS = True
    SNLI_TOTAL_SAMPLES_LIMIT = 20000
    SNLI_SAMPLES_PER_EPOCH = 50000

    
    # CNN feature paramters
    EMBEDDING_PATH = "GoogleNews-vectors-negative300.bin"
    CNN_INPUT_LENGTH = 250
    EMBEDDING_DIM = 300
    FILTER_SIZES = [2, 3, 4]
    NUM_FILTERS = 128

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
    BATCH_SIZE_TRAIN = 66
    EPOCHS = 100

    with open(TRAINING_LOG_FILE, 'w') as f:
        print("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_CNN_FEATURES, USE_AVG_EMBEDDINGS")
        print(USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_CNN_FEATURES, USE_AVG_EMBEDDINGS)
        print("USE_SNLI_NEUTRAL, SNLI_TOTAL_SAMPLES_LIMIT, SNLI_SAMPLES_PER_EPOCH")
        print(USE_SNLI_NEUTRAL, SNLI_TOTAL_SAMPLES_LIMIT, SNLI_SAMPLES_PER_EPOCH)

        f.write("USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_CNN_FEATURES, USE_AVG_EMBEDDINGS\n")
        vals = [USE_DOMAINS, ADD_FEATURES_TO_LABEL_PRED, USE_RELATIONAL_FEATURE_VECTORS, USE_CNN_FEATURES, USE_AVG_EMBEDDINGS]
        f.write(', '.join(str(val) for val in vals) + "\n")
        f.write("USE_SNLI_NEUTRAL, SNLI_TOTAL_SAMPLES_LIMIT, SNLI_SAMPLES_PER_EPOCH\n")
        vals = [USE_SNLI_NEUTRAL, SNLI_TOTAL_SAMPLES_LIMIT, SNLI_SAMPLES_PER_EPOCH]
        f.write(', '.join(str(val) for val in vals) + "\n")

        ### Extract Data ###

        print("Getting Data...")
        f.write("Getting Data...\n")
        
        if USE_FNC_DATA:
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
            fnc_domain = [0 for i in range(len(fnc_headlines_train))]
        else:
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_domain = [], [], [], []
        
        fnc_headlines_test, fnc_bodies_test, fnc_labels_test = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)

        snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=SNLI_TOTAL_SAMPLES_LIMIT, use_neutral=USE_SNLI_NEUTRAL)
        snli_domain = [1 for i in range(len(snli_s1_train))]

        train_headlines = fnc_headlines_train + snli_s1_train
        train_bodies = fnc_bodies_train + snli_s2_train
        train_labels = fnc_labels_train + snli_labels_train
        train_domains = fnc_domain + snli_domain
        test_headlines = fnc_headlines_test
        test_bodies = fnc_bodies_test
        test_labels = fnc_labels_test
        test_domains = [0 for _ in range(len(test_headlines))]
        
        assert len(fnc_headlines_train) == len(fnc_bodies_train) == len(fnc_labels_train) == len(fnc_domain)
        assert len(snli_s1_train) == len(snli_s2_train) == len(snli_labels_train) == len(snli_domain)
        assert len(test_headlines) == len(test_labels) == len(fnc_labels_test)

        ### Get TF and TFIDF vectorizers ###

        print("Creating Vectorizers...")
        f.write("Creating Vectorizers...\n")

        train_data = fnc_headlines_train + snli_s1_train + fnc_bodies_train + snli_s2_train
        test_data = fnc_headlines_test + fnc_bodies_test

        if ONLY_VECT_FNC:
            train_data = fnc_headlines_train + fnc_bodies_train
            test_data = fnc_headlines_test + fnc_bodies_test

        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(train_data, test_data, MAX_FEATURES)

        ### Get Feature Vectors ###
        print("Getting Feature Vectors...")
        f.write("Getting Feature Vectors...\n")
        print("  FNC train...")
        f.write("  FNC train...\n")
        fnc_train_vectors = get_feature_vectors(fnc_headlines_train, fnc_bodies_train, 
                                                bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)   
        f.write("    Number of Feature Vectors: " + str(len(fnc_train_vectors)) + "\n")

        print("  SNLI train...")
        f.write("  SNLI train...\n")
        snli_train_vectors = get_feature_vectors(snli_s1_train, snli_s2_train,
                                                 bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, use_cache=False)
        f.write("    Number of Feature Vectors: " + str(len(snli_train_vectors)) + "\n")
        
        train_vectors = fnc_train_vectors + snli_train_vectors
        print("  FNC test...")
        f.write("  FNC test...\n")
        test_vectors = get_feature_vectors(test_headlines, test_bodies,
                                           bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
        f.write("    Number of Feature Vectors: " + str(len(test_vectors)) + "\n")
        
        if USE_RELATIONAL_FEATURE_VECTORS:
            print("Getting Relational Feature Vectors...")
            f.write("Getting Relational Feature Vectors...\n")
            print("  train...")
            f.write("  train...\n")
            train_relational_vectors = get_relational_feature_vectors(train_vectors)
            print("  test...")
            f.write("  test...\n")
            test_relational_vectors = get_relational_feature_vectors(test_vectors)

        # Two BOW representations and single value for TFIDF representation
        FEATURE_VECTOR_SIZE = MAX_FEATURES * 2 + 1

        if USE_CNN_FEATURES or USE_AVG_EMBEDDINGS:
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
            train_headline_avg_embeddings = get_average_embeddings(train_headlines, embeddings)
            train_body_avg_embeddings = get_average_embeddings(train_bodies, embeddings)
            test_headline_avg_embeddings = get_average_embeddings(test_headlines, embeddings)
            test_body_avg_embeddings = get_average_embeddings(test_bodies, embeddings)

        if USE_CNN_FEATURES:
            print("Extracting CNN Inputs...")
            f.write("Extracting CNN Inputs...\n")

            print("  Tokenizing Text...")
            f.write("  Tokenizing Text...\n")
            # Map words to ids
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_headlines + train_bodies)
            train_headline_sequences = tokenizer.texts_to_sequences(train_headlines)
            train_body_sequences = tokenizer.texts_to_sequences(train_bodies)
            test_headline_sequences = tokenizer.texts_to_sequences(test_headlines)
            test_body_sequences = tokenizer.texts_to_sequences(test_bodies)
            word_index = tokenizer.word_index

            print("  Padding Sequences...")
            f.write("  Padding Sequences...\n")
            # truncate all inputs to CNN_INPUT_LENGTH
            x_train_headlines = pad_sequences(train_headline_sequences, maxlen=CNN_INPUT_LENGTH)
            x_train_bodies = pad_sequences(train_body_sequences, maxlen=CNN_INPUT_LENGTH)
            x_test_headlines = pad_sequences(test_headline_sequences, maxlen=CNN_INPUT_LENGTH)
            x_test_bodies = pad_sequences(test_body_sequences, maxlen=CNN_INPUT_LENGTH)
 
            print("  Creating Embedding Matrix...")
            f.write("  Creating Embedding Matrix...\n")
            # Create embedding matrix for embedding layer. Matrix will be 
            # (num_words + 1) x EMBEDDING_DIM since word_index starts at 1
            num_words = len(word_index)
            embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
            for word, rank in word_index.items():
                if word in embeddings.vocab:
                    embedding_matrix[rank] = embeddings[word]

        print("Defining Model...")
        f.write("Defining Model...\n")
        
        if USE_AVG_EMBEDDINGS:
            avg_embeddings_headline_pl = tf.placeholder(tf.float32, [None, EMBEDDING_DIM])
            avg_embeddings_body_pl = tf.placeholder(tf.float32, [None, EMBEDDING_DIM])

        # Create placeholders
        if USE_CNN_FEATURES:
            embedding_matrix_pl = tf.placeholder(tf.float32, [len(word_index) + 1, EMBEDDING_DIM])
            W = tf.Variable(tf.constant(0.0, shape=[len(word_index) + 1, EMBEDDING_DIM]), trainable=False)
            embedding_init = W.assign(embedding_matrix_pl)
            headline_words_pl = tf.placeholder(tf.int64, [None, CNN_INPUT_LENGTH])
            body_words_pl = tf.placeholder(tf.int64, [None, CNN_INPUT_LENGTH])

        features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE])
        stances_pl = tf.placeholder(tf.int64, [None])
        p_features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE])
        keep_prob_pl = tf.placeholder(tf.float32)
        domains_pl = tf.placeholder(tf.int64, [None])
        gr_pl = tf.placeholder(tf.float32, [])
        lr_pl = tf.placeholder(tf.float32, [])

        batch_size = tf.shape(features_pl)[0]
        
        ### Feature Extraction ###
        
        if USE_AVG_EMBEDDINGS:
            hidden_layer = tf.concat([avg_embeddings_headline_pl, avg_embeddings_body_pl], 1)

        elif USE_CNN_FEATURES:

            # CNN output fully connected to hidden layer of HIDDEN_SIZE
            pooled_outputs = []
            
            headline_embeddings = tf.nn.embedding_lookup(embedding_init, headline_words_pl)
            body_embeddings = tf.nn.embedding_lookup(embedding_init, body_words_pl)

            # Get headline features
            for filter_size in FILTER_SIZES:
                b_head = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
                conv_head = tf.layers.conv1d(headline_embeddings, NUM_FILTERS, filter_size)
                relu_head = tf.nn.relu(tf.nn.bias_add(conv_head, b_head))
                pool_head = tf.layers.max_pooling1d(conv_head, CNN_INPUT_LENGTH - filter_size + 1, 1)
                pooled_outputs.append(pool_head)

            # Get body features
            for filter_size in FILTER_SIZES:
                b_body = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
                conv_body = tf.layers.conv1d(body_embeddings, NUM_FILTERS, filter_size)
                relu_body = tf.nn.relu(tf.nn.bias_add(conv_body, b_head))
                pool_body = tf.layers.max_pooling1d(conv_body, CNN_INPUT_LENGTH - filter_size + 1, 1)
                pooled_outputs.append(pool_body)
            
            # Concat features and connect to hidden layer
            cnn_out_vector = tf.reshape(tf.concat(pooled_outputs, 2), [-1, NUM_FILTERS * len(FILTER_SIZES) * 2])
            cnn_out_vector = tf.nn.dropout(cnn_out_vector, keep_prob_pl)
            hidden_layer = tf.layers.dense(cnn_out_vector, HIDDEN_SIZE)
        
        else:
            # TF and TFIDF features fully connected hidden layer of HIDDEN_SIZE
            hidden_layer = tf.nn.dropout(tf.nn.relu(tf.layers.dense(features_pl, HIDDEN_SIZE)), keep_prob=keep_prob_pl)

        ### Label Prediction ###

        # Fully connected hidden layer with size based on LABEL_HIDDEN_SIZE with original features concated
        if ADD_FEATURES_TO_LABEL_PRED:
            hidden_layer_p = tf.concat([p_features_pl, hidden_layer], axis=1)

        if LABEL_HIDDEN_SIZE is None:
            hidden_layer_p = hidden_layer
        else:
            hidden_layer_p = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer, LABEL_HIDDEN_SIZE)), keep_prob=keep_prob_pl)

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

        # # Load model
        # if MODE == 'load':
        #    with tf.Session() as sess:
        #        saver = tf.train.Saver()
        #        saver.restore(sess, LOAD_MODEL_PATH)
        #
        #        # Predict
        #        test_feed_dict = {features_pl: test_vectors,
        #                          keep_prob_pl: 1.0,
        #                          domains_pl: [0 for i in range(len(test_vectors))],
        #                          gr_pl: 1.0}
        #        test_pred = sess.run(predict, feed_dict=test_feed_dict)

        # Train model
        if MODE == 'train':

            print("Training Model...")
            f.write("Training Model...\n")

            # Define optimiser
            opt_func = tf.train.AdamOptimizer(lr_pl)
            grads, _ = tf.clip_by_global_norm(tf.gradients(p_loss + d_loss + l2_loss, tf_vars), CLIP_RATIO)
            opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
            
            test_feed_dict = {features_pl: test_vectors,
                              p_features_pl: test_vectors,
                              stances_pl: test_labels,
                              keep_prob_pl: 1.0,
                              domains_pl: test_domains,
                              gr_pl: 1.0}

            if USE_RELATIONAL_FEATURE_VECTORS:
                test_feed_dict = {features_pl: test_relational_vectors,
                                  p_features_pl: test_vectors,
                                  stances_pl: test_labels,
                                  keep_prob_pl: 1.0,
                                  domains_pl: test_domains,
                                  gr_pl: 1.0}

            if USE_AVG_EMBEDDINGS:
                test_feed_dict = {avg_embeddings_headline_pl: test_headline_avg_embeddings,
                                  avg_embeddings_body_pl: test_body_avg_embeddings,
                                  features_pl: test_vectors,
                                  p_features_pl: test_vectors,
                                  stances_pl: test_labels,
                                  keep_prob_pl: 1.0,
                                  domains_pl: test_domains,
                                  gr_pl: 1.0}
            
            elif USE_CNN_FEATURES:
                test_feed_dict = {embedding_matrix_pl: embedding_matrix,
                                  headline_words_pl: x_test_headlines,
                                  body_words_pl: x_test_bodies,
                                  features_pl: test_vectors,
                                  p_features_pl: test_vectors,
                                  stances_pl: test_labels,
                                  keep_prob_pl: 1.0,
                                  domains_pl: test_domains,
                                  gr_pl: 1.0}

            best_loss = float('Inf')
            
            # Perform training
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
               
                for epoch in range(EPOCHS):
                    print("\n  EPOCH", epoch)
                    f.write("\n  EPOCH " + str(epoch))
                    
                    # Adaption Parameter and Learning Rate
                    p = float(epoch)/EPOCHS
                    gr = 2. / (1. + np.exp(-10. * p)) - 1
                    lr = 0.01 / (1. + 10 * p)**0.75
                
                    total_loss, total_p_loss, total_d_loss, total_reg_loss = 0, 0, 0, 0
                    train_l_pred = []
                    train_d_pred = []
                    
                    # Randomize order of FNC and SNLI data and randomly choose SNLI_SAMPLES_PER_EPOCH SNLI data
                    n_train = len(train_vectors)
                    n_fnc_train = len(fnc_headlines_train)
                    fnc_indices = list(range(n_fnc_train))
                    snli_indices = list(range(n_fnc_train, n_train))
                    rand1.shuffle(snli_indices)

                    if SNLI_SAMPLES_PER_EPOCH is not None:
                        indices = fnc_indices + snli_indices[:SNLI_SAMPLES_PER_EPOCH]
                    else:
                        indices = fnc_indices + snli_indices
                    rand2.shuffle(indices)
                    
                    # Training epoch loop
                    for i in range(len(indices) // BATCH_SIZE_TRAIN):

                        # Get training batches
                        batch_indices = indices[i * BATCH_SIZE_TRAIN: (i + 1) * BATCH_SIZE_TRAIN]
                        batch_features = [train_vectors[i] for i in batch_indices]
                        batch_stances = [train_labels[i] for i in batch_indices]
                        batch_domains = [train_domains[i] for i in batch_indices]

                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relational_features = [train_relational_vectors[i] for i in batch_indices]
                        
                        if USE_AVG_EMBEDDINGS:
                            batch_headline_avg_embeddings = [train_headline_avg_embeddings[i] for i in batch_indices]
                            batch_body_avg_embeddings = [train_body_avg_embeddings[i] for i in batch_indices]

                        if USE_CNN_FEATURES:
                            batch_headlines = [x_train_headlines[i] for i in batch_indices]
                            batch_bodies = [x_train_bodies[i] for i in batch_indices]

                        batch_feed_dict = {features_pl: batch_features,
                                           p_features_pl: batch_features,
                                           stances_pl: batch_stances,
                                           keep_prob_pl: TRAIN_KEEP_PROB,
                                           domains_pl: batch_domains,
                                           gr_pl: gr,
                                           lr_pl: lr}

                        if USE_RELATIONAL_FEATURE_VECTORS:
                            batch_feed_dict = {features_pl: batch_relational_features,
                                               p_features_pl: batch_features,
                                               stances_pl: batch_stances,
                                               keep_prob_pl: TRAIN_KEEP_PROB,
                                               domains_pl: batch_domains,
                                               gr_pl: gr,
                                               lr_pl: lr}
                        if USE_AVG_EMBEDDINGS:
                            batch_feed_dict = {avg_embeddings_headline_pl: batch_headline_avg_embeddings,
                                               avg_embeddings_body_pl: batch_body_avg_embeddings,
                                               features_pl: batch_features,
                                               p_features_pl: batch_features,
                                               stances_pl: batch_stances,
                                               keep_prob_pl: TRAIN_KEEP_PROB,
                                               domains_pl: batch_domains,
                                               gr_pl: gr,
                                               lr_pl: lr}
             
                        elif USE_CNN_FEATURES:
                            batch_feed_dict = {embedding_matrix_pl: embedding_matrix,
                                               headline_words_pl: batch_headlines,
                                               body_words_pl: batch_bodies,
                                               features_pl: batch_features,
                                               p_features_pl: batch_features,
                                               stances_pl: batch_stances,
                                               keep_prob_pl: TRAIN_KEEP_PROB,
                                               domains_pl: batch_domains,
                                               gr_pl: gr,
                                               lr_pl: lr}
                        
                        # Loss and predictions for each batch
                        _, lpred, dpred, ploss, dloss, l2loss = sess.run([opt_op, predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)
                        
                        # Total loss for current epoch
                        total_p_loss += ploss
                        total_d_loss += dloss
                        total_reg_loss += l2loss
                        total_loss += ploss + dloss + l2loss
                        train_l_pred.extend(lpred)
                        if USE_DOMAINS:
                            train_d_pred.extend(dpred)

                    
                    # Record loss and accuracy information for train
                    print("    Train Label Loss =", total_p_loss)
                    print("    Train Domain Loss =", total_d_loss)
                    print("    Train Regularization Loss =", total_reg_loss)
                    print("    Train Total Loss =", total_loss)

                    f.write("    Train Label Loss = " + str(total_p_loss) + "\n")
                    f.write("    Train Domain Loss = " + str(total_d_loss) + "\n")
                    f.write("    Train Regularization Loss = " + str(total_reg_loss) + "\n")
                    f.write("    Train Total Loss = " + str(total_loss) + "\n")

                    train_labels_epoch = [train_labels[i] for i in indices]
                    pred_accuracies = get_prediction_accuracies(train_l_pred, train_labels_epoch, 4)
                    print("    Train Label Accuracy", pred_accuracies)
                    f.write("    Train Label Accuracy [" + ', '.join(str(acc) for acc in pred_accuracies) + "]\n")

                    if USE_DOMAINS:
                        train_domain_labels_epoch = [train_domains[i] for i in indices]
                        domain_accuracies = get_prediction_accuracies(train_d_pred, train_domain_labels_epoch, 2)
                        print("    Train Domain Accuracy", domain_accuracies)
                        f.write("    Train Domain Accuracy [" + ', '.join(str(acc) for acc in domain_accuracies) + "]\n")

                    # Record loss and accuracy information for test
                    test_pred, test_d_pred, test_p_loss, test_d_loss, test_l2_loss = sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=test_feed_dict)

                    print("\n    Test Label Loss =", test_p_loss)
                    print("    Test Domain Loss =", test_d_loss)
                    print("    Test Regularization Loss =", test_l2_loss)
                    print("    Test Total Loss =", test_p_loss + test_d_loss + test_l2_loss)
 
                    f.write("    Test Label Loss = " + str(test_p_loss) + "\n")
                    f.write("    Test Domain Loss = " + str(test_d_loss) + "\n")
                    f.write("    Test Regularization Loss = " + str(test_l2_loss) + "\n")
                    f.write("    Test Total Loss = " + str(test_p_loss + test_d_loss + test_l2_loss) + "\n")
                    
                    # Save best test label loss model
                    if test_p_loss < best_loss:
                        best_loss = test_p_loss
                        saver = tf.train.Saver()
                        saver.save(sess, SAVE_MODEL_PATH)
                        print("    New Best Training Loss")
                        f.write("    New Best Training Loss\n")

                    composite_score = get_composite_score(test_pred, test_labels)
                    print("    Test Composite Score", composite_score)
                    f.write("    Test Composite Score " + str(composite_score) + "\n")
                    
                    pred_accuracies = get_prediction_accuracies(test_pred, test_labels, 4)
                    print("    Test Label Accuracy", pred_accuracies)
                    f.write("    Test Label Accuracy [" + ', '.join(str(acc) for acc in pred_accuracies) + "]\n")
                    
                    if USE_DOMAINS:
                        test_domain_labels = [0 for _ in range(len(test_labels))]
                        domain_accuracies = get_prediction_accuracies(test_d_pred, test_domain_labels, 2)
                        print("    Test Domain Accuracy", domain_accuracies)
                        f.write("    Test Domain Accuracy [" + ', '.join(str(acc) for acc in domain_accuracies) + "]\n")

        # Save predictions
        save_predictions(test_pred, test_labels, PREDICTION_FILE)


if __name__ == "__main__":
    main()

