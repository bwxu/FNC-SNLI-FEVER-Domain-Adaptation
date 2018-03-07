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
import nltk

FNC_LABELS = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
FNC_LABELS_REV = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
SNLI_LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
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
    skipped = 0
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i - skipped >= limit:
                break
            data = json.loads(line)
            label = data['gold_label']
            if label == "neutral" and not use_neutral:
                skipped += 1
                continue
            s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
            s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                skipped += 1
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
    '''
    Convert data into feature vectors where the first NUM_FEATURES elements is the 
    TF vector for the first document and the next NUM_FEATURES elements is the TF
    vector for the second document. The cosine distance between the TFIDF values
    of the vectors are then appended to this vector.

    The output will be feature_vectors, a len(data) x (2*NUM_FEATURES + 1) vector
    '''
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


def get_relational_feature_vectors(feature_vectors):
    '''
    Create relational feature vectors where the first NUM_FEATURES elements are
    the square of the difference between the feature vectors of the TF values for
    the two documents at that corresponding vertex. The next NUM_FEATURES elements
    represent the product of the two corresponding TF values. The value of the 
    cosine distance of the TFIDF values are then appended to this vector.

    Expected input is the list of feature_vectors from get_feature_vectors

    Expected output is a len(feature_vectors) x (2*NUM_FEATURES + 1) list
    '''
    # Calculate number of features per document tf vector
    NUM_FEATURES = len(feature_vectors[0])//2

    relational_feature_vectors = []

    for i in range(len(feature_vectors)):
        if (i % 5000) == 0:
            print("    Processed", i, "out of", len(feature_vectors))

        current_vector = feature_vectors[i]
        dist_vector = [(current_vector[j] - current_vector[5000+j])**2 for j in range(NUM_FEATURES)]
        mag_vector = [current_vector[j] * current_vector[5000+j] for j in range(NUM_FEATURES)]
        relational_vector = dist_vector + mag_vector + [current_vector[10000]]
        relational_feature_vectors.append(np.asarray(relational_vector))

    print("    Number of Relational Feature Vectors:", len(relational_feature_vectors))

    return relational_feature_vectors

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

def get_composite_score(pred, labels):
    score = 0         
    for i in range(len(pred)):
        # Unrelated label
        if labels[i] == FNC_LABELS['unrelated'] and pred[i] == FNC_LABELS['unrelated']:
            score += 0.25

        # Related label
        if labels[i] != FNC_LABELS['unrelated'] and pred[i] != FNC_LABELS['unrelated']:
            score += 0.25
            if labels[i] == pred[i]:
                score += 0.75
    return score

def get_prediction_accuracies(pred, labels, num_labels):
    correct = [0 for _ in range(num_labels)]
    total = [0 for _ in range(num_labels)]
    
    for i in range(len(pred)):
        total[labels[i]] += 1
        if pred[i] == labels[i]:
            correct[labels[i]] += 1

    # Avoid dividing by 0 case
    for label in range(len(total)):
        if total[label] == 0:
            total[label] += 1

    return [correct[i]/total[i] for i in range(len(total))]

def remove_stop_words(sentences):
    stops = set(stopwords.words('english'))
    sentences = [[word for word in nltk.word_tokenize(sentence.lower()) if word not in STOP_WORDS] for sentence in sentences]
    sentences = [' '.join(word for word in sentence) for sentence in sentences]
    return sentences

def get_average_embeddings(sentences, embeddings, embedding_size=300):
    '''
    Given a list of texts, and word2vec embeddings, produce a 300 length
    avg embedding vector for each text
    '''
    sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    avg_embeddings = [np.asarray([0 for _ in range(embedding_size)]) for _ in range(len(sentences))]
    for i, sentence in enumerate(sentences):
        for word in sentence:
            if word in embeddings.vocab:
                np.add(avg_embeddings[i], embeddings[word])
        avg_embeddings[i] = np.divide(avg_embeddings[i], float(len(sentence)))

    return avg_embeddings

