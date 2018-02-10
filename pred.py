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

from util import get_fnc_data, get_snli_data, get_vectorizers, get_feature_vectors  

def main():
    # Mode is either train or load
    MODE = 'train'
    TRAIN_SETS = ['FNC', 'SNLI']

    # File paths
    FNC_TRAIN_STANCES = "fnc_data/train_stances.csv"
    FNC_TRAIN_BODIES = "fnc_data/train_bodies.csv"
    FNC_TEST_STANCES = "fnc_data/competition_test_stances.csv"
    FNC_TEST_BODIES = "fnc_data/competition_test_bodies.csv"

    SNLI_TRAIN = 'snli_data/snli_1.0_train.jsonl' 
    SNLI_VAL = 'snli_data/snli_1.0_dev.jsonl'
    SNLI_TEST = 'snli_data/snli_1.0_test.jsonl'

    PREDICTION_FILE = 'test_preds.csv'

    # Model parameters
    r = random.Random()
    MAX_FEATURES = 5000
    TARGET_SIZE = 4
    HIDDEN_SIZE = 100
    TRAIN_KEEP_PROB = 0.6
    L2_ALPHA = 0.00001
    LEARN_RATE = 0.01
    CLIP_RATIO = 5
    BATCH_SIZE_TRAIN = 500
    EPOCHS = 50
    
    print("Getting Data...")

    fnc_headlines_train, fnc_bodies_train, fnc_labels_train = get_fnc_data(FNC_TRAIN_STANCES, FNC_TRAIN_BODIES)
    fnc_headlines_test, fnc_bodies_test, fnc_labels_test = get_fnc_data(FNC_TEST_STANCES, FNC_TEST_BODIES)

    snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(SNLI_TRAIN, limit=0)
    snli_s1_val, snli_s2_val, snli_labels_val = get_snli_data(SNLI_VAL)
    snli_s1_test, snli_s2_test, snli_labels_test = get_snli_data(SNLI_TEST)

    train_headlines = fnc_headlines_train + snli_s1_train
    train_bodies = fnc_bodies_train + snli_s2_train
    train_labels = fnc_labels_train + snli_labels_train
    test_headlines = fnc_headlines_test
    test_bodies = fnc_bodies_test
    test_labels = fnc_labels_test
    
    assert len(train_headlines) == len(train_bodies)
    assert len(test_headlines) == len(test_bodies)
    assert len(train_headlines) == len(train_labels)
    assert len(test_headlines) == len(test_labels)

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
    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SIZE], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, HIDDEN_SIZE)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, TARGET_SIZE), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, TARGET_SIZE])

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * L2_ALPHA

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(softmaxed_logits, axis=1)

    # Load model
    if MODE == 'load':
        with tf.Session() as sess:
            load_model(sess)


            # Predict
            test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
            test_pred = sess.run(predict, feed_dict=test_feed_dict)


    # Train model
    if MODE == 'train':

        print("Training Model...")

        # Define optimiser
        opt_func = tf.train.AdamOptimizer(LEARN_RATE)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), CLIP_RATIO)
        opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
        
        test_feed_dict = {features_pl: test_vectors, keep_prob_pl: 1.0}
        n_train = len(train_vectors)
        best_loss = float('Inf')
        
        # Perform training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(EPOCHS):
                print("  EPOCH", epoch)

                total_loss = 0
                indices = list(range(n_train))
                r.shuffle(indices)

                for i in range(n_train // BATCH_SIZE_TRAIN):
                    batch_indices = indices[i * BATCH_SIZE_TRAIN: (i + 1) * BATCH_SIZE_TRAIN]
                    batch_features = [train_vectors[i] for i in batch_indices]
                    batch_stances = [train_labels[i] for i in batch_indices]

                    batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: TRAIN_KEEP_PROB}
                    _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                    total_loss += current_loss

                print("    Epoch Loss =", total_loss)
                if total_loss < best_loss:
                    best_loss = total_loss
                    print("    New Best Training Loss")
                
                test_pred = sess.run(predict, feed_dict=test_feed_dict)
                
                correct = [0, 0, 0, 0]
                total = [0, 0, 0, 0]
                score = 0
                
                for i in range(len(test_pred)):
                    total[test_labels[i]] += 1
                    if test_pred[i] == test_labels[i]:
                        correct[test_labels[i]] += 1
                    
                    # Unrelated label
                    if test_labels[i] == 3 and test_pred[i] == 3:
                        score += 0.25

                    # Related label
                    if test_labels[i] != 3 and test_pred[i] != 3:
                        score += 0.25
                        if test_labels[i] == test_pred[i]:
                            score += 0.75

                print("    Composite Score", score)
                print("    Label Accuracy", [correct[i]/total[i] for i in range(len(total))])
     
    # Save predictions
    save_predictions(test_pred, test_labels, PREDICTION_FILE)


if __name__ == "__main__":
    main()

