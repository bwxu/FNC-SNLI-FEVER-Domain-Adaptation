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
from keras.utils import np_utils
import numpy as np

from flip_gradient import flip_gradient 
from util import get_fnc_data, get_snli_data, get_vectorizers, get_feature_vectors, save_predictions, get_prediction_accuracies, get_composite_score  

def main():
    # Mode is either train or load
    MODE = 'train'
    
    # Train and load parameters
    LOAD_MODEL_PATH = ""
    SAVE_FOLDER = "models/dann_snli/"
    PREDICTION_FILE = SAVE_FOLDER + "dann_snli.csv"
    SAVE_MODEL_PATH = SAVE_FOLDER + "dann_snli"
    USE_SNLI_DATA = True
    USE_SNLI_NEUTRAL = False
   
    # File paths
    FNC_TRAIN_STANCES = "fnc_data/train_stances.csv"
    FNC_TRAIN_BODIES = "fnc_data/train_bodies.csv"
    FNC_TEST_STANCES = "fnc_data/competition_test_stances.csv"
    FNC_TEST_BODIES = "fnc_data/competition_test_bodies.csv"

    SNLI_TRAIN = 'snli_data/snli_1.0_train.jsonl' 
    SNLI_VAL = 'snli_data/snli_1.0_dev.jsonl'
    SNLI_TEST = 'snli_data/snli_1.0_test.jsonl'

    # Model parameters
    r = random.Random()
    MAX_FEATURES = 5000
    TARGET_SIZE = 4
    DOMAIN_TARGET_SIZE = 2
    HIDDEN_SIZE = 100
    DOMAIN_HIDDEN_SIZE = 10
    LABEL_HIDDEN_SIZE = 10
    TRAIN_KEEP_PROB = 0.6
    L2_ALPHA = 0.01
    CLIP_RATIO = 5
    BATCH_SIZE_TRAIN = 500
    EPOCHS = 90
   
    print("Getting Data...")

    fnc_headlines_train, fnc_bodies_train, fnc_labels_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
    fnc_domain = [0 for i in range(len(fnc_headlines_train))]
    
    fnc_headlines_test, fnc_bodies_test, fnc_labels_test = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)

    # Control whether to use SNLI training data
    LIMIT = None
    if not USE_SNLI_DATA:
        LIMIT = 0

    snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=20000, use_neutral=USE_SNLI_NEUTRAL)
    snli_domain = [1 for i in range(len(snli_s1_train))]
    #snli_s1_val, snli_s2_val, snli_labels_val = get_snli_data(SNLI_VAL)
    #snli_s1_test, snli_s2_test, snli_labels_test = get_snli_data(SNLI_TEST)

    train_headlines = fnc_headlines_train + snli_s1_train
    train_bodies = fnc_bodies_train + snli_s2_train
    train_labels = fnc_labels_train + snli_labels_train
    train_domains = fnc_domain + snli_domain
    test_headlines = fnc_headlines_test
    test_bodies = fnc_bodies_test
    test_labels = fnc_labels_test
    
    assert len(train_headlines) == len(train_bodies)
    assert len(test_headlines) == len(test_bodies)
    assert len(train_headlines) == len(train_labels)
    assert len(test_headlines) == len(test_labels)
    assert len(train_headlines) == len(train_domains)

    print("Creating Vectorizers...")

    train_data = train_headlines + train_bodies
    test_data = test_headlines + test_bodies
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(train_data, test_data, MAX_FEATURES)

    print("Getting Feature Vectors...")
    print("  FNC train...")
    fnc_train_vectors = get_feature_vectors(fnc_headlines_train, fnc_bodies_train, 
                                            bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    print("  SNLI train...")
    snli_train_vectors = get_feature_vectors(snli_s1_train, snli_s2_train,
                                             bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, use_cache=False)
    train_vectors = fnc_train_vectors + snli_train_vectors
    print("  FNC test...")
    test_vectors = get_feature_vectors(test_headlines, test_bodies,
                                       bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    
    # Two BOW representations and single value for TFIDF representation
    FEATURE_VECTOR_SIZE = MAX_FEATURES * 2 + 1

    print("Defining Model...")
    
    ### Feature Extraction ###

    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)
    domains_pl = tf.placeholder(tf.int64, [None], 'domains')
    gr_pl = tf.placeholder(tf.float32, [], 'gr')
    lr_pl = tf.placeholder(tf.float32, [], 'lr')

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, HIDDEN_SIZE)), keep_prob=keep_prob_pl)

    ### Label Prediction ###
    if LABEL_HIDDEN_SIZE is None:
        hidden_layer_p = hidden_layer
    else:
        hidden_layer_p = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer, LABEL_HIDDEN_SIZE)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer_p, TARGET_SIZE), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, TARGET_SIZE])

    # Define overall loss
    p_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl))

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(softmaxed_logits, axis=1)

    ### Domain Prediction ###
    hidden_layer_d = flip_gradient(hidden_layer, gr_pl)
    if DOMAIN_HIDDEN_SIZE is None:
        domain_layer = hidden_layer_d
    else:
        domain_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer_d, DOMAIN_HIDDEN_SIZE)), keep_prob=keep_prob_pl)
    d_logits_flat = tf.nn.dropout(tf.contrib.layers.linear(domain_layer, DOMAIN_TARGET_SIZE), keep_prob=keep_prob_pl)
    d_logits = tf.reshape(d_logits_flat, [batch_size, DOMAIN_TARGET_SIZE])

    d_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_logits, labels=domains_pl))

    softmaxed_d_logits = tf.nn.softmax(d_logits)
    d_predict = tf.argmax(softmaxed_d_logits, axis=1)

    ### Regularization ###
    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * L2_ALPHA

    # Load model
    if MODE == 'load':
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, LOAD_MODEL_PATH)

            # Predict
            test_feed_dict = {features_pl: test_vectors,
                              keep_prob_pl: 1.0,
                              domains_pl: [0 for i in range(len(test_vectors))],
                              gr_pl: 1.0}
            test_pred = sess.run(predict, feed_dict=test_feed_dict)

    # Train model
    if MODE == 'train':

        print("Training Model...")

        # Define optimiser
        opt_func = tf.train.AdamOptimizer(lr_pl)
        grads, _ = tf.clip_by_global_norm(tf.gradients(p_loss + d_loss + l2_loss, tf_vars), CLIP_RATIO)
        opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
        
        test_feed_dict = {features_pl: test_vectors,
                          stances_pl: test_labels,
                          keep_prob_pl: 1.0,
                          domains_pl: [0 for i in range(len(test_vectors))],
                          gr_pl: 1.0}

        n_train = len(train_vectors)
        best_loss = float('Inf')
        
        # Perform training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
           
            for epoch in range(EPOCHS):
                print("\n  EPOCH", epoch)
                
                # Adaption Parameter and Learning Rate
                p = float(epoch)/EPOCHS
                gr = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p)**0.75
            
                total_loss = 0
                total_p_loss = 0
                total_d_loss = 0
                total_reg_loss = 0
                train_l_pred = []
                train_d_pred = []

                indices = list(range(n_train))
                r.shuffle(indices)

                for i in range(n_train // BATCH_SIZE_TRAIN):
                    batch_indices = indices[i * BATCH_SIZE_TRAIN: (i + 1) * BATCH_SIZE_TRAIN]
                    batch_features = [train_vectors[i] for i in batch_indices]
                    batch_stances = [train_labels[i] for i in batch_indices]
                    batch_domains = [train_domains[i] for i in batch_indices]

                    batch_feed_dict = {features_pl: batch_features,
                                       stances_pl: batch_stances,
                                       keep_prob_pl: TRAIN_KEEP_PROB,
                                       domains_pl: batch_domains,
                                       gr_pl: gr,
                                       lr_pl: lr}

                    _, lpred, dpred, ploss, dloss, l2loss = sess.run([opt_op, predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)
                    
                    total_p_loss += ploss
                    total_d_loss += dloss
                    total_reg_loss = l2loss
                    total_loss += ploss + dloss + l2loss
                    train_l_pred.extend(lpred)
                    train_d_pred.extend(dpred)

                print("    Train Label Loss =", total_p_loss)
                print("    Train Domain Loss =", total_d_loss)
                print("    Train Regularization Loss =", total_reg_loss)
                print("    Train Total Loss =", total_loss)
                
                train_labels_epoch = [train_labels[i] for i in indices]
                pred_accuracies = get_prediction_accuracies(train_l_pred, train_labels_epoch, 4)
                print("    Train Label Accuracy", pred_accuracies)
                train_domain_labels_epoch = [train_domains[i] for i in indices]
                domain_accuracies = get_prediction_accuracies(train_d_pred, train_domain_labels_epoch, 2)
                print("    Train Domain Accuracy", domain_accuracies)

                test_pred, test_d_pred, test_p_loss, test_d_loss, test_l2_loss = sess.run([predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=test_feed_dict)

                print("\n    Test Label Loss =", test_p_loss)
                print("    Test Domain Loss =", test_d_loss)
                print("    Test Regularization Loss =", test_l2_loss)
                print("    Test Total Loss =", test_p_loss + test_d_loss + test_l2_loss)
                
                if test_p_loss < best_loss:
                    best_loss = test_p_loss
                    print("    New Best Training Loss")
                    
                    # save the model
                    saver = tf.train.Saver()
                    saver.save(sess, SAVE_MODEL_PATH)

                composite_score = get_composite_score(test_pred, test_labels)
                print("    Test Composite Score", composite_score)
                pred_accuracies = get_prediction_accuracies(test_pred, test_labels, 4)
                print("    Test Label Accuracy", pred_accuracies)
                test_domain_labels = [0 for _ in range(len(test_labels))]
                domain_accuracies = get_prediction_accuracies(test_d_pred, test_domain_labels, 2)
                print("    Test Domain Accuracy", domain_accuracies)

    # Save predictions
    save_predictions(test_pred, test_labels, PREDICTION_FILE)


if __name__ == "__main__":
    main()

