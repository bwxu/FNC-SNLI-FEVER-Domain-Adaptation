import numpy as np
import tensorflow as tf
import var
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from util import print_model_results, get_prediction_accuracies, get_composite_score, \
        get_f1_scores, save_predictions, get_feature_vectors, remove_stop_words, \
        get_body_sentences

claim = "There are 305 breeds of domestic rabbit in the world"
document = ["Rabbits are small mammals in the family Leporidae of the order Lagomorpha (along with the hare and the pika). Oryctolagus cuniculus includes the European rabbit species and its descendants, the world's 305 breeds of domestic rabbit. Sylvilagus includes thirteen wild rabbit species, among them the seven types of cottontail. The European rabbit, which has been introduced on every continent except Antarctica, is familiar throughout the world as a wild prey animal and as a domesticated form of livestock and pet. With its widespread effect on ecologies and cultures, the rabbit (or bunny) is, in many areas of the world, a part of daily life—as food, clothing, and companion, and as a source of artistic inspiration."]

bow_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "bow_vectorizer.pickle", "rb")) 
tfreq_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "tfreq_vectorizer.pickle", "rb"))
tfidf_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "tfidf_vectorizer.pickle", "rb"))

document_sentences = get_body_sentences(document)[0]
print(document_sentences)

claims = [claim for _ in range(len(document_sentences) + 1)]
documents = document + document_sentences

claim_doc_feat_vectors = get_feature_vectors(claims, documents, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

word_index = np.load(var.PICKLE_SAVE_FOLDER + "word_index.npy").item()

# initialize tokenizer from word_index
words = [(word, word_index[word]) for word in word_index]
sorted(words, key=lambda x: x[1])
sentence = " ".join([word[0] for word in words])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([sentence])

miss = 0
for key in word_index:
    if word_index[key] != tokenizer.word_index[key]:
        miss += 1

assert miss == 0

claims = remove_stop_words(claims)
documents = remove_stop_words(documents)

claim_seqs = tokenizer.texts_to_sequences(claims)
document_seqs = tokenizer.texts_to_sequences(documents)

claim_seqs_padded = pad_sequences(claim_seqs, maxlen=var.CNN_HEADLINE_LENGTH)
document_seqs_padded = pad_sequences(document_seqs, maxlen=var.CNN_BODY_LENGTH)

stances = [0 for _ in range(len(documents))]
domains = [0 for _ in range(len(documents))]
indicies = [i for i in range(len(documents))]

if var.USE_CNN_FEATURES:
    embedding_matrix = np.load(
        var.PICKLE_SAVE_FOLDER + "embedding_matrix.npy")
    word_index = np.load(
        var.PICKLE_SAVE_FOLDER + "word_index.npy")
    word_index = word_index.item()

graph1 = tf.Graph()
graph2 = tf.Graph()

related_unrelated_preds = []
related_unrelated_logits = []
three_label_preds = []
three_label_logits = []

with tf.Session(graph=graph2) as sess:
    saver = tf.train.import_meta_graph(var.RELATED_UNRELATED_MODEL_PATH + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint(var.RELATED_UNRELATED_MODEL_FOLDER))

    graph = tf.get_default_graph()
       
    features_pl = graph.get_tensor_by_name("features_pl:0")
    stances_pl = graph.get_tensor_by_name("stances_pl:0")
    keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")
    domains_pl = graph.get_tensor_by_name("domains_pl:0")
    gr_pl = graph.get_tensor_by_name("gr_pl:0")
    lr_pl = graph.get_tensor_by_name("lr_pl:0")
    
    p_loss = graph.get_tensor_by_name("p_loss:0")
    p_predict = graph.get_tensor_by_name("p_predict:0")
    p_logits = graph.get_tensor_by_name("Softmax:0")
    d_loss = graph.get_tensor_by_name("d_loss:0")
    d_predict = graph.get_tensor_by_name("d_predict:0")
    l2_loss = graph.get_tensor_by_name("l2_loss:0")

    for i in range(len(documents) // var.BATCH_SIZE + 1):
        batch_indices = indicies[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
        batch_stances = [stances[i] for i in batch_indices]
        batch_features = [claim_doc_feat_vectors[i] for i in batch_indices]
        
        batch_feed_dict = {
            stances_pl: batch_stances,
            features_pl: batch_features,
            keep_prob_pl: 1.0
        }

        # Record loss and accuracy information for test
        lpred, dpred, ploss, plogits, dloss, l2loss = \
            sess.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                      l2_loss], feed_dict=batch_feed_dict)

        related_unrelated_preds.extend(lpred)
        related_unrelated_logits.extend(plogits)

    var.FNC_LABELS_REV[0] = "related"
    for i in range(len(related_unrelated_preds)):
        print("", documents[i][:60] + " ...")
        print("    label: ", var.FNC_LABELS_REV[related_unrelated_preds[i]])
        print("    logits: ", related_unrelated_logits[i][0], related_unrelated_logits[i][3])
    var.FNC_LABELS_REV[0] = "agree"


with tf.Session(graph=graph1) as sess:
    # Load 3 label model
    print(var.THREE_LABEL_MODEL_PATH)
    print(var.THREE_LABEL_MODEL_FOLDER)
    saver = tf.train.import_meta_graph(var.THREE_LABEL_MODEL_PATH + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint(var.THREE_LABEL_MODEL_FOLDER))

    graph = tf.get_default_graph()
    
    if var.USE_TF_VECTORS or var.USE_RELATIONAL_FEATURE_VECTORS or var.USE_AVG_EMBEDDINGS:
        features_pl = graph.get_tensor_by_name("features_pl:0")
    
    if var.USE_CNN_FEATURES:
        embedding_matrix_pl = graph.get_tensor_by_name("embedding_matrix_pl:0")
        headline_words_pl = graph.get_tensor_by_name("headline_words_pl:0")
        body_words_pl = graph.get_tensor_by_name("body_words_pl:0")

    if var.ADD_FEATURES_TO_LABEL_PRED:
        p_features_pl = graph.get_tensor_by_name("p_features_pl:0")

    stances_pl = graph.get_tensor_by_name("stances_pl:0")
    keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")
    domains_pl = graph.get_tensor_by_name("domains_pl:0")
    gr_pl = graph.get_tensor_by_name("gr_pl:0")
    lr_pl = graph.get_tensor_by_name("lr_pl:0")
    
    p_loss = graph.get_tensor_by_name("p_loss:0")
    p_predict = graph.get_tensor_by_name("p_predict:0")
    p_logits = graph.get_tensor_by_name("Softmax:0")
    d_loss = graph.get_tensor_by_name("d_loss:0")
    d_predict = graph.get_tensor_by_name("d_predict:0")
    l2_loss = graph.get_tensor_by_name("l2_loss:0")

    for i in range(len(documents) // var.BATCH_SIZE + 1):
        batch_indices = indicies[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
        batch_stances = [stances[i] for i in batch_indices]
        batch_domains = [domains[i] for i in batch_indices]
        batch_features = [claim_doc_feat_vectors[i] for i in batch_indices]
        
        batch_feed_dict = {stances_pl: batch_stances,
                           keep_prob_pl: 1.0}

        if var.USE_DOMAINS:
            batch_feed_dict[gr_pl] = 1.0
            batch_feed_dict[domains_pl] = batch_domains

        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
            if var.USE_TF_VECTORS:
                batch_feed_dict[features_pl] = batch_features
            if var.ADD_FEATURES_TO_LABEL_PRED:
                batch_feed_dict[p_features_pl] = batch_features

        if var.USE_CNN_FEATURES:
            batch_x_headlines = [claim_seqs_padded[i] for i in batch_indices]
            batch_x_bodies = [document_seqs_padded[i] for i in batch_indices]
            batch_feed_dict[headline_words_pl] = batch_x_headlines
            batch_feed_dict[body_words_pl] = batch_x_bodies
            batch_feed_dict[embedding_matrix_pl] = embedding_matrix

        # Record loss and accuracy information for test
        lpred, dpred, ploss, plogits, dloss, l2loss = \
            sess.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                      l2_loss], feed_dict=batch_feed_dict)

        three_label_preds.extend(lpred)
        three_label_logits.extend(plogits)
    
    for i in range(len(three_label_preds)):
        print("", documents[i][:60] + " ...")
        print("    label:", var.FNC_LABELS_REV[three_label_preds[i]])
        print("    logits: ", three_label_logits[i][0], three_label_logits[i][1], three_label_logits[i][2])



