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
SNLI_LABELS = {'entailment': 0, 'contradiction': 1}
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


def get_snli_examples(jsonl_path, skip_no_majority=True, limit=None, use_neutral=True):
    examples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i == limit:
                break
            data = json.loads(line)
            label = data['gold_label']
            if label == "neutral" and not use_neutral:
                continue
            s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
            s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                continue
            examples.append((label, s1, s2))
    return examples


def get_snli_data(jsonl_path, limit=None, use_neutral=True):
    data = get_snli_examples(jsonl_path=jsonl_path, limit=limit, use_neutral=use_neutral)
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
    
    print("    Number of Feature Vectors:", len(feature_vectors))

    return feature_vectors

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
            writer.writerow({'Stance': FNC_LABELS_REV[pred[i]], 'Actual': FNC_LABELS_REV[actual[i]]})


