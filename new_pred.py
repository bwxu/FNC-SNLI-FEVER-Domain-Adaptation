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


# Import relevant packages and modules
from util import *
import random
import tensorflow as tf
import json
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  #print(max(len(x.split()) for x in left))
  #print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y


# Prompt for mode
mode = 'train'

# Set file names
file_train_instances = "fnc/train_stances.csv"
file_train_bodies = "fnc/train_bodies.csv"
file_test_instances = "fnc/test_stances_unlabeled.csv"
file_test_bodies = "fnc/test_bodies.csv"
file_predictions = 'snli_test_predictions.csv'

file_snli_train = 'snli_1.0/snli_1.0_train.jsonl' 
file_snli_val = 'snli_1.0/snli_1.0_dev.jsonl'
file_snli_test = 'snli_1.0/snli_1.0_test.jsonl'

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 3
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90

# Load data sets
#raw_train = FNCData(file_train_instances, file_train_bodies)
#raw_test = FNCData(file_test_instances, file_test_bodies)
#n_train = len(raw_train.instances)

snli_train_s1, snli_train_s2, snli_train_label = get_data(file_snli_train)
print()
print(snli_train_s1[:10])
print(snli_train_s2[:10])
print(snli_train_label[:10])
n_train = len(snli_train_s1)

train_stances = [np.nonzero(label == 1)[0][0] for label in snli_train_label]
print("Train Stances", train_stances[:10])

snli_val_s1, snli_val_s2, snli_val_label = get_data(file_snli_val)
print()
print(snli_val_s1[:10])
print(snli_val_s2[:10])
print(snli_val_label[:10])

snli_test_s1, snli_test_s2, snli_test_label = get_data(file_snli_test) 
print()
print(snli_test_s1[:10])
print(snli_test_s2[:10])
print(snli_test_label[:10])
test_stances = [np.nonzero(label == 1)[0][0] for label in snli_test_label]
print("Test Stances", test_stances[:10])

# Process data sets
#train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
#    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
#feature_size = len(train_set[0])
#test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

print('hi')
train_set, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = snli_pipeline_train(snli_train_s1, snli_train_s2, snli_test_s1, snli_test_s2, lim_unigram)
test_set = snli_pipeline_test(snli_test_s1, snli_test_s2, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
feature_size = len(train_set[0])
# Define model

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)


# Load model
if mode == 'load':
    with tf.Session() as sess:
        load_model(sess)


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Train model
if mode == 'train':

    print("Training Model")

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            print("EPOCH", epoch)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss

            test_pred = sess.run(predict, feed_dict=test_feed_dict)

            correct = 0.0
            wrong = 0.0
            for i in range(len(test_pred)):
                if test_pred[i] != test_stances[i]:
                    wrong += 1
                else:
                    correct += 1
            print("Correct", correct)
            print("Wrong", wrong)
            print("Accuracy", correct/(correct + wrong))
 
# Save predictions
save_predictions(test_pred, file_predictions)

print(len(test_pred))
print(len(test_stances))


