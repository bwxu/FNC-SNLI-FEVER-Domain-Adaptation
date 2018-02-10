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

from csv import DictReader, DictWriter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import numpy as np
import tensorflow as tf


FNC_LABELS = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
FNC_LABELS_REV = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
SNLI_LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
SNLI_LABELS_REV = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
STOP_WORDS = set(stopwords.words('english'))


def open_csv(path):
    with open(path, "r", encoding='utf-8') as table:
        rows = [row for row in DictReader(table)]
        return rows


def get_fnc_data(stances_path, bodies_path):
    stances_file = open_csv(stances_path)
    bodies_file = open_csv(bodies_path)

    headlines = [row['Headline'] for row in stances_file]
    body_id_to_article = {int(row['Body ID']): row['articleBody'] for row in bodies_file} 
    bodies = [body_id_to_article[int(row['Body ID'])] for row in stances_file]
    labels = [FNC_LABELS[row['Stance']] for row in stances_file]
    
    return headlines, bodies, labels


def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()


def get_snli_examples(jsonl_path, skip_no_majority=True, limit=None):
    examples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit and i > limit:
                break
            data = json.loads(line)
            label = data['gold_label']
            s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
            s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                continue
            examples.append((label, s1, s2))
    return examples


def get_snli_data(jsonl_path, limit=None):
    data = get_snli_examples(jsonl_path=jsonl_path, limit=limit)
    left = [s1 for _, s1, _ in data]
    right = [s2 for _, _, s2 in data]
    labels = [SNLI_LABELS[l] for l, _, _ in data]
    return left, right, labels


def get_vectorizers(train_data, test_data, MAX_FEATURES):
    train_data = list(set(train_data))
    test_data = list(set(test_data))
    
    bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words=STOP_WORDS)
    bow = bow_vectorizer.fit_transform(train_data)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)

    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=STOP_WORDS).\
        fit(train_data + test_data)  # Train and test sets

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def get_feature_vectors(headlines, bodies, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, use_cache=True):
    feature_vectors = []
    headline_cache = {}
    body_cache = {}

    for i in range(len(headlines)):
        if i % 5000 == 0:
            print("    Processed", i, "out of", len(headlines))
        
        headline = headlines[i]
        body = bodies[i]
       
        if use_cache and headline in headline_cache:
            headline_tf = headline_cache[headline][0]
            headline_tfidf = headline_cache[headline][1]

        else:
            headline_bow = bow_vectorizer.transform([headline]).toarray()
            headline_tf = tfreq_vectorizer.transform(headline_bow).toarray()[0].reshape(1, -1)
            headline_tfidf = tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
            if use_cache:
                headline_cache[headline] = (headline_tf, headline_tfidf)
        
        if use_cache and body in body_cache:
            body_tf = body_cache[body][0]
            body_tfidf = body_cache[body][1]
        
        else:
            body_bow = bow_vectorizer.transform([body]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
            if use_cache:
                body_cache[body] = (body_tf, body_tfidf)

        tfidf_cos = cosine_similarity(headline_tfidf, body_tfidf)[0].reshape(1, 1)
        feature_vector = np.squeeze(np.c_[headline_tf, body_tf, tfidf_cos])
        feature_vectors.append(feature_vector)
    
    print("    Number of Feature Vectors: ", len(feature_vectors))

    return feature_vectors


def load_model(sess):

    """

    Load TensorFlow model

    Args:
        sess: TensorFlow session

    """

    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')


def save_predictions(pred, actual, file):

    """

    Save predictions to CSV file

    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension

    """
    
    with open(file, 'w') as csvfile:
        fieldnames = ['Stance', 'Actual']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(pred)):
            writer.writerow({'Stance': snli_label_ref_rev[pred[i]], 'Actual': snli_label_ref_rev[actual[i]]})



# Define relevant functions
def pipeline_train(train, test, lim_unigram):

    """

    Process train set, create relevant vectorizers

    Args:
        train: FNCData object, train set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    train_stances = []
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_tfidf_track = {}
    body_tfidf_track = {}

    # Identify unique heads and bodies
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            heads.append(head)
            heads_track[head] = 1
        if body_id not in bodies_track:
            bodies.append(train.bodies[body_id])
            bodies_track[body_id] = 1
            body_ids.append(body_id)

    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in test_heads_track:
            test_heads.append(head)
            test_heads_track[head] = 1
        if body_id not in test_bodies_track:
            test_bodies.append(test.bodies[body_id])
            test_bodies_track[body_id] = 1
            test_body_ids.append(body_id)

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
        fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

    # Process train set
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        head_tf = tfreq[id_ref[head]].reshape(1, -1)
        body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
        if head not in head_tfidf_track:
            head_tfidf = tfidf_vectorizer.transform([head]).toarray()
            head_tfidf_track[head] = head_tfidf
        else:
            head_tfidf = head_tfidf_track[head]
        if body_id not in body_tfidf_track:
            body_tfidf = tfidf_vectorizer.transform([train.bodies[body_id]]).toarray()
            body_tfidf_track[body_id] = body_tfidf
        else:
            body_tfidf = body_tfidf_track[body_id]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        train_set.append(feat_vec)
        train_stances.append(label_ref[instance['Stance']])

    return train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer

def snli_pipeline_train(train_s1, train_s2, test_s1, test_s2, lim_unigram):
    num_train = len(train_s1)
    print()
    print("Num training samples", num_train)
    num_test = len(test_s1)

    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(train_s1 + train_s2)

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()
    print("Num lines of text, len bow", len(tfreq), len(tfreq[0]))

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
        fit(train_s1 + train_s2 + test_s1 + test_s2)

    train_set = []

    for i in range(num_train):
        s1_tfidf = tfidf_vectorizer.transform([train_s1[i]]).toarray()
        s2_tfidf = tfidf_vectorizer.transform([train_s2[i]]).toarray()
        tfidf_cos = cosine_similarity(s1_tfidf, s2_tfidf)[0].reshape(1, 1)

        s1_tf = tfreq[i].reshape(1, -1)
        s2_tf = tfreq[i + num_train].reshape(1, -1)

        feat_vec = np.squeeze(np.c_[s1_tf, s2_tf, tfidf_cos])
        if i % 1000 == 0:
            print(i, "out of", num_train)
        train_set.append(feat_vec)

    print("Train set dims", len(train_set), len(train_set[0]))

    return train_set, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):

    """

    Process test set

    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        test_set: list, of numpy arrays

    """

    # Initialise
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process test set
    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            head_bow = bow_vectorizer.transform([head]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
            heads_track[head] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[head][0]
            head_tfidf = heads_track[head][1]
        if body_id not in bodies_track:
            body_bow = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test_set.append(feat_vec)

    return test_set

def snli_pipeline_test(test_s1, test_s2, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    test_set = []
    for i in range(len(test_s1)):
        s1_tfidf = tfidf_vectorizer.transform([test_s1[i]]).toarray().reshape(1, -1)
        s2_tfidf = tfidf_vectorizer.transform([test_s2[i]]).toarray().reshape(1, -1)
        tfidf_cos = cosine_similarity(s1_tfidf, s2_tfidf)[0].reshape(1, 1)
        
        s1_bow = bow_vectorizer.transform([test_s1[i]]).toarray()
        s1_tf = tfreq_vectorizer.transform(s1_bow).toarray()[0].reshape(1, -1)

        s2_bow = bow_vectorizer.transform([test_s2[i]]).toarray()
        s2_tf = tfreq_vectorizer.transform(s2_bow).toarray()[0].reshape(1, -1)

        feat_vec = np.squeeze(np.c_[s1_tf, s2_tf, tfidf_cos])
        test_set.append(feat_vec)
    
    print("Test set dims", len(test_set), len(test_set[0]))

    return test_set






