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
import imp
import sys
sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
import nltk
import os

from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import numpy as np
import tensorflow as tf
import unicodedata

FNC_LABELS = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
FNC_LABELS_REV = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
SNLI_LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


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
    body_ids = [int(row['Body ID']) for row in stances_file]
    
    return headlines, bodies, labels, body_ids


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

def extract_fever_train_jsonl_data(train_path):
    num_train = 0
    total_ev = 0
    
    headlines = []
    labels = []
    article_list = []
    claim_set = set()

    with open(train_path, 'r') as f:
        for item in f:
            data = json.loads(item)
            claim_set.add(data["claim"])
            if data["verifiable"] == "VERIFIABLE":
                evidence_articles = set()
                for evidence in data["all_evidence"]:
                    article_name = unicodedata.normalize('NFC', evidence[2])
                    if article_name in evidence_articles:
                        continue
                    else:                    
                        article_list.append(article_name)
                        evidence_articles.add(article_name)
                        headlines.append(data["claim"])
                        labels.append(FEVER_LABELS[data["label"]])

                    total_ev += 1
                num_train += 1
    
    print("Num Distinct Claims", num_train)
    print("Num Data Points", total_ev)

    return headlines, labels, article_list, claim_set

def get_relevant_articles(wikidata_path, article_list):
    article_dict = {article: None for article in article_list}
    
    wiki_files = [os.path.join(wikidata_path, f) for f in os.listdir(wikidata_path)]
    print(wiki_files[:10])

    total_num_files = 0
    for file in wiki_files:
        print(file)
        with open(file, 'r') as f:
            for item in f:
                data = json.loads(item)
                key = unicodedata.normalize('NFC', data["id"])
                if key in article_dict:
                    article_dict[key] = data["text"]
                total_num_files += 1
                
    print("Total Num Wiki Articles", total_num_files)

    bodies = []
    for article in article_list:
        bodies.append(article_dict[article])
    
    return bodies

def get_fever_data(train_path, wikidata_path):
    headlines, labels, article_list, claim_set = extract_fever_train_jsonl_data(train_path)
    print(headlines[:20], article_list[:20])
    bodies = get_relevant_articles(wikidata_path, article_list)
    return headlines, bodies, labels, claim_set

def get_vectorizers(train_data, MAX_FEATURES):
    train_data = list(set(train_data))
    
    bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words=STOP_WORDS)
    bow = bow_vectorizer.fit_transform(train_data)

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)

    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=STOP_WORDS).fit(train_data)

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def get_feature_vectors(headlines, bodies, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    '''
    Convert data into feature vectors where the first NUM_FEATURES elements is the 
    TF vector for the first document and the next NUM_FEATURES elements is the TF
    vector for the second document. The cosine distance between the TFIDF values
    of the vectors are then appended to this vector.

    The output will be feature_vectors, a len(data) x (2*NUM_FEATURES + 1) vector
    '''
    feature_vectors = []

    for i in range(len(headlines)):
        if i % 5000 == 0:
            print("    Processed", i, "out of", len(headlines))
        
        headline = headlines[i]
        body = bodies[i]
       
        headline_bow = bow_vectorizer.transform([headline]).toarray()
        headline_tf = tfreq_vectorizer.transform(headline_bow).toarray()[0].reshape(1, -1)
        headline_tfidf = tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
        
        body_bow = bow_vectorizer.transform([body]).toarray()
        body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
        body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)

        tfidf_cos = cosine_similarity(headline_tfidf, body_tfidf)[0].reshape(1, 1)
        feature_vector = np.squeeze(np.c_[headline_tf, body_tf, tfidf_cos])
        
        feature_vectors.append(feature_vector)

    print("    Number of Feature Vectors:", len(feature_vectors))
    
    feature_vectors = np.asarray(feature_vectors)

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

    relational_feature_vectors = np.zeros((len(feature_vectors), 10001))

    for i in range(len(feature_vectors)):
        if (i % 5000) == 0:
            print("    Processed", i, "out of", len(feature_vectors))
        
        tf_vector_1 = feature_vectors[i][:NUM_FEATURES]
        tf_vector_2 = feature_vectors[i][NUM_FEATURES:2*NUM_FEATURES]
        tfidf = feature_vectors[i][2*NUM_FEATURES:]
        
        dist_vector = np.power(tf_vector_1 - tf_vector_2, 2)
        mag_vector = np.multiply(tf_vector_1, tf_vector_2)

        relational_vector = np.concatenate([dist_vector, mag_vector, tfidf])

        relational_feature_vectors[i] = relational_vector

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
    sentences = [[word for word in nltk.word_tokenize(sentence.lower()) if word not in STOP_WORDS] for sentence in sentences]
    sentences = [' '.join(word for word in sentence) for sentence in sentences]
    return sentences

def get_average_embeddings(sentences, embeddings, embedding_size=300):
    '''
    Given a list of texts, and word2vec embeddings, produce a 300 length
    avg embedding vector for each text
    '''
    sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    avg_embeddings = np.zeros((len(sentences), embedding_size))
    
    for i, sentence in enumerate(sentences):
        if len(sentence) == 0:
            continue

        if i % 5000 == 0:
            print("    Processed", i, "out of", len(sentences))

        count = 0.0
        for word in sentence:
            if word in embeddings.vocab:
                count += 1
                avg_embeddings[i] += embeddings[word]
        if count > 0:
            avg_embeddings[i] /= count
    
    return avg_embeddings
                   
def print_model_results(f, set_name, pred, labels, d_pred, d_labels, p_loss, d_loss, l2_loss, USE_DOMAINS):
    print("\n    " + set_name + "  Label Loss =", p_loss)
    print("    " + set_name + "  Domain Loss =", d_loss)
    print("    " + set_name + "  Regularization Loss =", l2_loss)
    print("    " + set_name + "  Total Loss =", p_loss + d_loss + l2_loss)

    f.write("\n    " + set_name + "  Label Loss = " + str(p_loss) + "\n")
    f.write("    " + set_name + "  Domain Loss = " + str(d_loss) + "\n")
    f.write("    " + set_name + "  Regularization Loss = " + str(l2_loss) + "\n")
    f.write("    " + set_name + "  Total Loss = " + str(p_loss + d_loss + l2_loss) + "\n")
    
    composite_score = get_composite_score(pred, labels)
    print("    " + set_name + "  Composite Score", composite_score)
    f.write("    " + set_name + "  Composite Score " + str(composite_score) + "\n")
    
    pred_accuracies = get_prediction_accuracies(pred, labels, 4)
    print("    " + set_name + "  Label Accuracy", pred_accuracies)
    f.write("    " + set_name + "  Label Accuracy [" + ', '.join(str(acc) for acc in pred_accuracies) + "]\n")
    
    if USE_DOMAINS:
        domain_accuracies = get_prediction_accuracies(d_pred, d_labels, 3)
        print("    " + set_name + "  Domain Accuracy", domain_accuracies)
        f.write("    " + set_name + "  Domain Accuracy [" + ', '.join(str(acc) for acc in domain_accuracies) + "]\n")

def get_body_sentences(bodies, flatten=False):
    result = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for body in bodies: 
        sents = tokenizer.tokenize(body)
        if flatten:
            result.extend(sents)
        else:
            result.append(sents)
    return result

def select_best_body_sentences(headlines, bodies, tfidf_vectorizer):
    
    best_sents = []

    for i, headline in enumerate(headlines):
        if i % 1000 == 0:
            print('Finished ' + str(i) + ' out of ' + str(len(headlines)) + ' headlines')

        best_sent = None
        best_tfidf = -1
        
        for j, body_sent in enumerate(bodies[i]):
            headline_tfidf = tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([body_sent]).toarray().reshape(1, -1)
            
            tfidf_cos = cosine_similarity(headline_tfidf, body_tfidf)[0].reshape(1, 1)

            if tfidf_cos > best_tfidf:
                best_tfidf = tfidf_cos
                best_sent = body_sent

        best_sents.append(best_sent)

    return best_sents

def remove_data_with_label(labels_to_remove, headlines, bodies, labels, domains, additional=None):
    
    throwaway_indices = [i for i, x in enumerate(labels) if x in labels_to_remove]
    
    for i in sorted(throwaway_indices, reverse=True):
        del headlines[i]
        del bodies[i]
        del labels[i]
        del domains[i]
        if additional is not None:
            del additional[i]
    
    result = [headlines, bodies, labels, domains]
    if additional is not None:
        result.append(additional)
    return result


