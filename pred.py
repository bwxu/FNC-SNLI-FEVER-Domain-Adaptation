import math
import random
import tensorflow as tf
import numpy as np
import os
import var

from shutil import copyfile
from gensim.models.keyedvectors import KeyedVectors
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from flip_gradient import flip_gradient
from util import get_fnc_data, get_snli_data, get_fever_data, \
        get_vectorizers, get_feature_vectors, save_predictions, \
        get_relational_feature_vectors, remove_stop_words, \
        get_average_embeddings, print_model_results, \
        remove_data_with_label


def process_data():

    if not os.path.exists(var.PICKLE_SAVE_FOLDER):
        raise Exception("Specified PICKLE_SAVE_FOLDER doesn't exist")

    with open(var.PICKLE_LOG_FILE, 'w') as f:
        # SAVE PARAMS
        copyfile('var.py', os.path.join(var.PICKLE_SAVE_FOLDER, 'var.py'))

        # EXTRACT DATA
        print("Getting Data...")
        f.write("Getting Data...\n")

        # Choose labels to ignore
        LABELS_TO_IGNORE = set()
        if not var.USE_UNRELATED_LABEL:
            LABELS_TO_IGNORE.add(3)
        if not var.USE_DISCUSS_LABEL:
            LABELS_TO_IGNORE.add(2)

        # Initalize the train and validation lists
        train_headlines = []
        train_bodies = []
        train_labels = []
        train_domains = []
        val_headlines = []
        val_bodies = []
        val_labels = []
        val_domains = []
        train_sizes = {}
        val_sizes = {}

        # The VAL_SIZE_CAP is used to ensure that there's an equal number
        # of elements from each dataset in the validation set. Datasets are
        # examined from smallest to largest FNC -> FEVER -> SNLI
        VAL_SIZE_CAP = None

        # If FNC data is used, extract the relevant training, validation,
        # and test data
        if var.USE_FNC_DATA:
            print("EXTRACTING FNC DATA")
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_body_ids_train = get_fnc_data(
                var.FNC_TRAIN_STANCES, var.FNC_TRAIN_BODIES)
            fnc_headlines_test, fnc_bodies_test, fnc_labels_test, _ = get_fnc_data(
                var.FNC_TEST_STANCES, var.FNC_TEST_BODIES)
            fnc_domains_train = [0 for i in range(len(fnc_headlines_train))]
            fnc_domains_test = [0 for i in range(len(fnc_headlines_test))]

            # Remove unwanted labels determined by USE_UNRELATED_LABEL and
            # USE_DISCUSS_LABEL
            fnc_headlines_train, fnc_bodies_train, fnc_labels_train, fnc_domains_train, fnc_body_ids_train = \
                remove_data_with_label(
                    LABELS_TO_IGNORE,
                    fnc_headlines_train,
                    fnc_bodies_train,
                    fnc_labels_train,
                    fnc_domains_train,
                    additional=fnc_body_ids_train)

            fnc_headlines_test, fnc_bodies_test, fnc_labels_test, fnc_domains_test = \
                remove_data_with_label(
                    LABELS_TO_IGNORE,
                    fnc_headlines_test,
                    fnc_bodies_test,
                    fnc_labels_test,
                    fnc_domains_test)

            # Seperate the train in to train and validation sets such that
            # no body article is in both the train and val sets.
            unique_body_ids = list(set(fnc_body_ids_train))
            indices = list(range(len(unique_body_ids)))
            var.RAND0.shuffle(indices)

            FNC_TRAIN_END = int(len(indices) * (1 - var.VALIDATION_SET_SIZE))
            
            train_body_ids = set(unique_body_ids[i] for i in indices[:FNC_TRAIN_END])
            val_body_ids = set(unique_body_ids[i] for i in indices[FNC_TRAIN_END:])

            train_indices = set(i for i in range(len(fnc_body_ids_train)) if fnc_body_ids_train[i] in train_body_ids)
            val_indices = set(i for i in range(len(fnc_body_ids_train)) if fnc_body_ids_train[i] in val_body_ids)

            # Add training and val data into the overall training and
            # val sets.
            val_headlines += [fnc_headlines_train[i] for i in val_indices]
            val_bodies += [fnc_bodies_train[i] for i in val_indices]
            val_labels += [fnc_labels_train[i] for i in val_indices]
            val_domains += [fnc_domains_train[i] for i in val_indices]

            train_headlines = [fnc_headlines_train[i] for i in train_indices]
            train_bodies = [fnc_bodies_train[i] for i in train_indices]
            train_labels = [fnc_labels_train[i] for i in train_indices]
            train_domains = [fnc_domains_train[i] for i in train_indices]

            train_sizes['fnc'] = len(train_indices)
            val_sizes['fnc'] = len(val_indices)

            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines) * var.EXTRA_SAMPLES_PER_EPOCH

        # If FEVER data is used, extract the relevant training, validation,
        # and test data
        if var.USE_FEVER_DATA:
            print("EXTRACTING FEVER DATA")
            fever_headlines, fever_bodies, fever_labels, fever_claim_set = \
                get_fever_data(var.FEVER_TRAIN, var.FEVER_WIKI)
            fever_domains = [2 for _ in range(len(fever_headlines))]

            # Remove unwanted labels determined by USE_UNRELATED_LABEL and
            # USE_DISCUSS_LABEL
            fever_headlines, fever_bodies, fever_labels, fever_domains = \
                remove_data_with_label(
                    LABELS_TO_IGNORE,
                    fever_headlines,
                    fever_bodies,
                    fever_labels,
                    fever_domains)

            # Seperate into training, validation, and test sets based on 
            # the claims
            claim_list = list(fever_claim_set)
            claim_indices = list(range(len(claim_list)))
            var.RAND1.shuffle(claim_indices)

            FEVER_VAL_SIZE = int(len(claim_indices) * var.VALIDATION_SET_SIZE)
            FEVER_TRAIN_END = len(claim_indices) - 2 * FEVER_VAL_SIZE
            FEVER_VAL_END = len(claim_indices) - FEVER_VAL_SIZE
            
            train_claims = set([claim_list[i] for i in claim_indices[:FEVER_TRAIN_END]])
            val_claims = set([claim_list[i] for i in claim_indices[FEVER_TRAIN_END:FEVER_VAL_END]])
            test_claims = set([claim_list[i] for i in claim_indices[FEVER_VAL_END:]])

            train_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in train_claims]
            val_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in val_claims]
            test_indices = [i for i in range(len(fever_headlines)) if fever_headlines[i] in test_claims]

            # Add training and val data into overall training and val lists
            train_headlines += [fever_headlines[i] for i in train_indices]
            train_bodies += [fever_bodies[i] for i in train_indices]
            train_labels += [fever_labels[i] for i in train_indices]
            train_domains += [fever_domains[i] for i in train_indices]

            val_headlines += [fever_headlines[i] for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [fever_bodies[i] for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [fever_labels[i] for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [fever_domains[i] for i in val_indices][:VAL_SIZE_CAP]

            fever_headlines_test = [fever_headlines[i] for i in test_indices]
            fever_bodies_test = [fever_bodies[i] for i in test_indices]
            fever_labels_test = [fever_labels[i] for i in test_indices]
            fever_domains_test = [fever_domains[i] for i in test_indices]

            train_sizes['fever'] = len(train_indices)
            val_sizes['fever'] = len(val_indices)

            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines)

        # If SNLI data is used, extract the relevant training, validation,
        # and test data
        if var.USE_SNLI_DATA:
            print("EXTRACTING SNLI DATA")
            snli_s1_train, snli_s2_train, snli_labels_train = get_snli_data(
                var.SNLI_TRAIN)
            snli_domains = [1 for _ in range(len(snli_s1_train))]

            # Remove unwanted labels determined by USE_UNRELATED_LABEL and
            # USE_DISCUSS_LABEL
            snli_s1_train, snli_s2_train, snli_labels_train, snli_domains = \
                remove_data_with_label(
                    LABELS_TO_IGNORE,
                    snli_s1_train,
                    snli_s2_train,
                    snli_labels_train,
                    snli_domains)

            # Seperate into training, val, and test data based on the 
            # sentences
            s2_list = list(set(snli_s2_train))
            s2_indices = list(range(len(s2_list)))
            var.RAND1.shuffle(s2_indices)

            SNLI_VAL_SIZE = int(len(s2_indices) * var.VALIDATION_SET_SIZE)
            SNLI_TRAIN_END = len(s2_indices) - 2 * SNLI_VAL_SIZE
            SNLI_VAL_END = len(s2_indices) - SNLI_VAL_SIZE

            train_s2 = set(s2_list[i] for i in s2_indices[:SNLI_TRAIN_END])
            val_s2 = set(s2_list[i] for i in s2_indices[SNLI_TRAIN_END:SNLI_VAL_END])
            test_s2 = set(s2_list[i] for i in s2_indices[SNLI_VAL_END:])

            train_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in train_s2]
            val_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in val_s2]
            test_indices = [i for i in range(len(snli_s2_train)) if snli_s2_train[i] in test_s2]

            # Add the train and val data to the overall train and val
            # lists
            train_headlines += [snli_s1_train[i] for i in train_indices]
            train_bodies += [snli_s2_train[i] for i in train_indices]
            train_labels += [snli_labels_train[i] for i in train_indices]
            train_domains += [snli_domains[i] for i in train_indices]

            val_headlines += [snli_s1_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [snli_s2_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [snli_labels_train[i] for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [snli_domains[i] for i in val_indices][:VAL_SIZE_CAP]

            snli_headlines_test = [snli_s1_train[i] for i in test_indices]
            snli_bodies_test = [snli_s2_train[i] for i in test_indices]
            snli_labels_test = [snli_labels_train[i] for i in test_indices]
            snli_domains_test = [snli_domains[i] for i in train_indices]

            train_sizes['snli'] = len(train_indices)
            val_sizes['snli'] = len(val_indices)

            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines)

        # Get and filter test data
        if var.TEST_DATASET == 'FNC':
            test_headlines = fnc_headlines_test
            test_bodies = fnc_bodies_test
            test_labels = fnc_labels_test
            test_domains = fnc_domains_test

        elif var.TEST_DATASET == 'FEVER':
            test_headlines = fever_headlines_test
            test_bodies = fever_bodies_test
            test_labels = fever_labels_test
            test_domains = fever_domains_test

        elif var.TEST_DATASET == 'FEVER':
            test_headlines = fever_headlines_test
            test_bodies = fever_bodies_test
            test_labels = fever_labels_test
            test_domains = fever_domains_test

        test_headlines, test_bodies, test_labels, test_domains = \
            remove_data_with_label(
                LABELS_TO_IGNORE,
                test_headlines,
                test_bodies,
                test_labels,
                test_domains)

        np.save(var.PICKLE_SAVE_FOLDER + "train_sizes.npy", train_sizes)
        np.save(var.PICKLE_SAVE_FOLDER + "val_sizes.npy", val_sizes)

        # Save the train, val, and test labels and domains
        np.save(
            var.PICKLE_SAVE_FOLDER + "train_labels.npy",
            np.asarray(train_labels))
        del train_labels
        np.save(
            var.PICKLE_SAVE_FOLDER + "train_domains.npy",
            np.asarray(train_domains))
        del train_domains

        np.save(
            var.PICKLE_SAVE_FOLDER + "val_labels.npy",
            np.asarray(val_labels))
        del val_labels
        np.save(
            var.PICKLE_SAVE_FOLDER + "val_domains.npy",
            np.asarray(val_domains))
        del val_domains

        np.save(
            var.PICKLE_SAVE_FOLDER +
            "test_labels.npy",
            np.asarray(test_labels))
        del test_labels
        np.save(
            var.PICKLE_SAVE_FOLDER +
            "test_domains.npy",
            np.asarray(test_domains))
        del test_domains

        # GET FEATURE VECTORS

        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
            print("Creating Vectorizers...")
            f.write("Creating Vectorizers...\n")

            vec_train_data = train_headlines + train_bodies

            # Only train vectorizers with FNC data
            if var.ONLY_VECT_FNC and var.USE_FNC_DATA:
                vec_train_data = train_headlines[:train_sizes['fnc']] + fnc_bodies_train[:train_sizes['fnc']]

            bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(vec_train_data, var.MAX_FEATURES)
            del vec_train_data
            
            print("Getting Feature Vectors...")
            f.write("Getting Feature Vectors...\n")

            print("  train...")
            f.write("  train...\n")
            train_tf_vectors = get_feature_vectors(train_headlines, train_bodies,
                                                   bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "train_tf_vectors.npy",
                train_tf_vectors)
            f.write("    Number of Feature Vectors: " +
                    str(len(train_tf_vectors)) + "\n")

            print("  val...")
            f.write("  val...\n")
            val_tf_vectors = get_feature_vectors(val_headlines, val_bodies,
                                                 bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(
                var.PICKLE_SAVE_FOLDER + "val_tf_vectors.npy",
                val_tf_vectors)
            f.write("    Number of Feature Vectors: " +
                    str(len(val_tf_vectors)) + "\n")

            print("  test...")
            f.write("  test...\n")
            test_tf_vectors = get_feature_vectors(test_headlines, test_bodies,
                                                  bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

            f.write("    Number of Feature Vectors: " +
                    str(len(test_tf_vectors)) + "\n")
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "test_tf_vectors.npy",
                test_tf_vectors)

        if var.USE_RELATIONAL_FEATURE_VECTORS:
            print("Getting Relational Feature Vectors...")
            f.write("Getting Relational Feature Vectors...\n")

            print("  train...")
            f.write("  train...\n")
            train_relation_vectors = get_relational_feature_vectors(
                train_tf_vectors)
            del train_tf_vectors
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "train_relation_vectors.npy",
                train_relation_vectors)
            del train_relation_vectors

            print("  val...")
            f.write("  val...\n")
            val_relation_vectors = get_relational_feature_vectors(
                val_tf_vectors)
            del val_tf_vectors
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "val_relation_vectors.npy",
                val_relation_vectors)
            del val_relation_vectors

            print("  test...")
            f.write("  test...\n")
            test_relation_vectors = get_relational_feature_vectors(
                test_tf_vectors)
            del test_tf_vectors
            test_relation_vectors = np.asarray(test_relation_vectors)
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "test_relation_vectors.npy",
                test_relation_vectors)
            del test_relation_vectors

        if var.USE_AVG_EMBEDDINGS or var.USE_CNN_FEATURES:
            print("Proessing Embedding Data")
            f.write("Processing Embedding Data...\n")

            print("  Getting pretrained embeddings...")
            f.write("  Getting pretrained embeddings...\n")
            embeddings = KeyedVectors.load_word2vec_format(
                var.EMBEDDING_PATH, binary=True)

            print("  Removing Stop Words...")
            f.write("  Removing Stop Words...\n")
            # Remove stopwords from training and test data
            train_headlines = remove_stop_words(train_headlines)
            train_bodies = remove_stop_words(train_bodies)

            val_headlines = remove_stop_words(val_headlines)
            val_bodies = remove_stop_words(val_bodies)

            test_headlines = remove_stop_words(test_headlines)
            test_bodies = remove_stop_words(test_bodies)

        if var.USE_AVG_EMBEDDINGS:
            print("Getting Average Embedding Vectors...")
            f.write("Getting Average Embedding Vectors...\n")

            print("  Train Headline...")
            f.write("  Train Headline...\n")
            train_headline_avg_embeddings = get_average_embeddings(
                train_headlines, embeddings)

            print("  Train Body...")
            f.write("  Train Body...\n")
            train_body_avg_embeddings = get_average_embeddings(
                train_bodies, embeddings)

            print("  Val Headline...")
            f.write("  Val Headline...\n")
            val_headline_avg_embeddings = get_average_embeddings(
                val_headlines, embeddings)

            print("  Val Body...")
            f.write("  Val Body...\n")
            val_body_avg_embeddings = get_average_embeddings(
                val_bodies, embeddings)

            print("  Test Headline...")
            f.write("  Test Headline...\n")
            test_headline_avg_embeddings = get_average_embeddings(
                test_headlines, embeddings)

            print("  Train Headline...")
            f.write("  Train Headline...\n")
            test_body_avg_embeddings = get_average_embeddings(
                test_bodies, embeddings)

            print("  Combining Train Vectors...")
            f.write("  Combining Train Vectors...\n")
            train_avg_embed_vectors = [np.concatenate([train_headline_avg_embeddings[i], train_body_avg_embeddings[i]])
                                       for i in range(len(train_headline_avg_embeddings))]
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "train_avg_embed_vectors.npy",
                train_avg_embed_vectors)
            del train_avg_embed_vectors

            print("  Combining Val Vectors...")
            f.write("  Combining Val Vectors...")
            val_avg_embed_vectors = [np.concatenate([val_headline_avg_embeddings[i], train_body_avg_embeddings[i]])
                                     for i in range(len(val_headline_avg_embeddings))]
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "val_avg_embed_vectors.npy",
                val_avg_embed_vectors)
            del val_avg_embed_vectors

            print("  Combining Test Vectors...")
            f.write("  Combining Test Vectors...\n")
            test_avg_embed_vectors = [np.concatenate([test_headline_avg_embeddings[i], test_body_avg_embeddings[i]])
                                      for i in range(len(test_headline_avg_embeddings))]
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "test_avg_embed_vectors.npy",
                test_avg_embed_vectors)
            del test_avg_embed_vectors

        if var.USE_CNN_FEATURES:
            print("Getting CNN Input Vectors...")
            f.write("Getting CNN Input Vectors...\n")

            print("  Tokenizing Text...")
            f.write("  Tokenizing Text...\n")
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_headlines + train_bodies)

            train_headline_seq = tokenizer.texts_to_sequences(train_headlines)
            train_body_seq = tokenizer.texts_to_sequences(train_bodies)

            val_headline_seq = tokenizer.texts_to_sequences(val_headlines)
            val_body_seq = tokenizer.texts_to_sequences(val_bodies)

            test_headline_seq = tokenizer.texts_to_sequences(test_headlines)
            test_body_seq = tokenizer.texts_to_sequences(test_bodies)

            word_index = tokenizer.word_index
            np.save(var.PICKLE_SAVE_FOLDER + "word_index.npy", word_index)

            print("  Padding Sequences...")
            f.write("  Padding Sequences...\n")
            x_train_headlines = pad_sequences(
                train_headline_seq, maxlen=var.CNN_HEADLINE_LENGTH)
            x_train_bodies = pad_sequences(
                train_body_seq, maxlen=var.CNN_BODY_LENGTH)

            x_val_headlines = pad_sequences(
                val_headline_seq, maxlen=var.CNN_HEADLINE_LENGTH)
            x_val_bodies = pad_sequences(val_body_seq, maxlen=var.CNN_BODY_LENGTH)

            x_test_headlines = pad_sequences(
                test_headline_seq, maxlen=var.CNN_HEADLINE_LENGTH)
            x_test_bodies = pad_sequences(
                test_body_seq, maxlen=var.CNN_BODY_LENGTH)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_train_headlines.npy",
                x_train_headlines)
            np.save(var.PICKLE_SAVE_FOLDER + "x_train_bodies.npy", x_train_bodies)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_val_headlines.npy",
                x_val_headlines)
            np.save(var.PICKLE_SAVE_FOLDER + "x_val_bodies.npy", x_val_bodies)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_test_headlines.npy",
                x_test_headlines)
            np.save(var.PICKLE_SAVE_FOLDER + "x_test_bodies.npy", x_test_bodies)

            print("  Creating Embedding Matrix...")
            f.write("  Creating Embedding Matrix...\n")
            num_words = len(word_index)
            embedding_matrix = np.zeros((num_words + 1, var.EMBEDDING_DIM))
            for word, rank in word_index.items():
                if word in embeddings.vocab:
                    embedding_matrix[rank] = embeddings[word]

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "embedding_matrix.npy",
                embedding_matrix)


def train_model():
    
    if not os.path.exists(var.SAVE_FOLDER):
        raise Exception("Specified SAVE_FOLDER doesn't exist")
    
    with open(var.TRAINING_LOG_FILE, 'w') as f:
        # SAVE PARAMS
        copyfile('var.py', os.path.join(var.SAVE_FOLDER, 'var.py'))

        # Take last VALIDATION_SET_SIZE PROPORTION of train set as validation
        # set
        print("Loading train vectors...")
        f.write("Loading train vectors...\n")
        train_sizes = np.load(var.PICKLE_SAVE_FOLDER + "train_sizes.npy")
        train_sizes = train_sizes.item()
        train_labels = np.load(var.PICKLE_SAVE_FOLDER + "train_labels.npy")
        train_domains = np.load(var.PICKLE_SAVE_FOLDER + "train_domains.npy")

        print("Loading val vectors...")
        f.write("Loading val vectors...\n")
        val_sizes = np.load(var.PICKLE_SAVE_FOLDER + "val_sizes.npy")
        val_sizes = val_sizes.item()
        val_labels = np.load(var.PICKLE_SAVE_FOLDER + "val_labels.npy")
        val_domains = np.load(var.PICKLE_SAVE_FOLDER + "val_domains.npy")

        print("Loading test vectors...")
        f.write("Loading test vectors...\n")
        test_labels = np.load(var.PICKLE_SAVE_FOLDER + "test_labels.npy")
        test_domains = np.load(var.PICKLE_SAVE_FOLDER + "test_domains.npy")

        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
            print("Loading TF vectors...")
            f.write("Loading TF vectors...\n")

            train_tf_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "train_tf_vectors.npy")
            val_tf_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "val_tf_vectors.npy")
            test_tf_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "test_tf_vectors.npy")

            SIZE_TRAIN = len(train_tf_vectors)
            SIZE_VAL = len(val_tf_vectors)
            SIZE_TEST = len(test_tf_vectors)

        if var.USE_RELATIONAL_FEATURE_VECTORS:
            print("Loading relation vectors...")
            f.write("Loading relation vectors...\n")

            train_relation_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "train_relation_vectors.npy")
            val_relation_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "val_relation_vectors.npy")
            test_relation_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "test_relation_vectors.npy")

            SIZE_TRAIN = len(train_relation_vectors)
            SIZE_VAL = len(val_relation_vectors)
            SIZE_TEST = len(test_relation_vectors)

        if var.USE_AVG_EMBEDDINGS:
            print("Loading avg embedding vectors...")
            f.write("Loading avg embeddings vectors...\n")

            train_avg_embed_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "train_avg_embed_vectors.npy")
            val_avg_embed_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "val_avg_embed_vectors.npy")
            test_avg_embed_vectors = np.load(
                var.PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy")

            SIZE_TRAIN = len(train_avg_embed_vectors)
            SIZE_VAL = len(val_avg_embed_vectors)
            SIZE_TEST = len(test_avg_embed_vectors)

        if var.USE_CNN_FEATURES:
            print("Loading CNN vectors...")
            f.write("Loading CNN vectors...\n")

            x_train_headlines = np.load(
                var.PICKLE_SAVE_FOLDER + "x_train_headlines.npy")
            x_train_bodies = np.load(
                var.PICKLE_SAVE_FOLDER + "x_train_bodies.npy")
            x_val_headlines = np.load(
                var.PICKLE_SAVE_FOLDER + "x_val_headlines.npy")
            x_val_bodies = np.load(
                var.PICKLE_SAVE_FOLDER + "x_val_bodies.npy")
            x_test_headlines = np.load(
                var.PICKLE_SAVE_FOLDER + "x_test_headlines.npy")
            x_test_bodies = np.load(
                var.PICKLE_SAVE_FOLDER + "x_test_bodies.npy")

            SIZE_TRAIN = len(x_train_headlines)
            SIZE_VAL = len(x_val_headlines)
            SIZE_TEST = len(x_test_headlines)

            embedding_matrix = np.load(
                var.PICKLE_SAVE_FOLDER + "embedding_matrix.npy")
            word_index = np.load(
                var.PICKLE_SAVE_FOLDER + "word_index.npy")
            word_index = word_index.item()

        print("SIZE_TRAIN = ", SIZE_TRAIN)
        f.write("SIZE_TRAIN = " + str(SIZE_TRAIN) + "\n")

        print("SIZE_VAL = ", SIZE_VAL)
        f.write("SIZE_VAL = " + str(SIZE_VAL) + "\n")

        print("SIZE_TEST = ", SIZE_TEST)
        f.write("SIZE_TEST = " + str(SIZE_TEST) + "\n")

        ################
        # DEFINE MODEL #
        ################

        best_loss = float('Inf')

        for model_num in range(var.NUM_MODELS_TO_TRAIN):
            print("Training model " + str(model_num))
            f.write("Training model " + str(model_num) + "\n")
            tf.reset_default_graph()

            print("Defining Model...")
            f.write("Defining Model...\n")
            if var.USE_TF_VECTORS:
                FEATURE_VECTOR_SIZE = len(train_tf_vectors[0])
            elif var.USE_RELATIONAL_FEATURE_VECTORS:
                FEATURE_VECTOR_SIZE = len(train_relation_vectors[0])
            elif var.USE_AVG_EMBEDDINGS:
                FEATURE_VECTOR_SIZE = len(train_avg_embed_vectors[0])
            elif var.USE_CNN_FEATURES:
                FEATURE_VECTOR_SIZE = var.NUM_FILTERS * len(var.FILTER_SIZES) * 2

            stances_pl = tf.placeholder(tf.int64, [None], name="stances_pl")
            keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob_pl")
            domains_pl = tf.placeholder(tf.int64, [None], name="domains_pl")
            gr_pl = tf.placeholder(tf.float32, [], name="gr_pl")
            lr_pl = tf.placeholder(tf.float32, [], name="lr_pl")

            if var.USE_TF_VECTORS or var.USE_RELATIONAL_FEATURE_VECTORS or var.USE_AVG_EMBEDDINGS:
                features_pl = tf.placeholder(
                    tf.float32, [
                        None, FEATURE_VECTOR_SIZE], name="features_pl")
                batch_size = tf.shape(features_pl)[0]

            if var.ADD_FEATURES_TO_LABEL_PRED:
                p_features_pl = tf.placeholder(
                    tf.float32, [
                        None, len(
                            train_tf_vectors[0])], name="p_features_pl")

            if var.USE_CNN_FEATURES:
                embedding_matrix_pl = tf.placeholder(
                    tf.float32, [len(word_index) + 1, var.EMBEDDING_DIM])
                W = tf.Variable(
                    tf.constant(0.0, shape=[len(word_index) + 1, var.EMBEDDING_DIM]),
                    trainable=False)
                embedding_init = W.assign(embedding_matrix_pl)
                headline_words_pl = tf.placeholder(
                    tf.int64, [None, len(x_train_headlines[0])])
                body_words_pl = tf.placeholder(
                    tf.int64, [None, len(x_train_bodies[0])])
                batch_size = tf.shape(headline_words_pl)[0]

            ### Feature Extraction ###

            # TF and TFIDF features fully connected hidden layer of HIDDEN_SIZE
            if var.USE_CNN_FEATURES:
                pooled_outputs = []

                headline_embeddings = tf.nn.embedding_lookup(
                    embedding_init, headline_words_pl)
                body_embeddings = tf.nn.embedding_lookup(
                    embedding_init, body_words_pl)

                for filter_size in var.FILTER_SIZES:
                    b_head = tf.Variable(tf.constant(0.1, shape=[var.NUM_FILTERS]))
                    conv_head = tf.layers.conv1d(
                        headline_embeddings, var.NUM_FILTERS, filter_size)
                    relu_head = tf.nn.relu(tf.nn.bias_add(conv_head, b_head))
                    pool_head = tf.layers.max_pooling1d(
                        relu_head, var.CNN_HEADLINE_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_head)

                for filter_size in var.FILTER_SIZES:
                    b_body = tf.Variable(tf.constant(0.1, shape=[var.NUM_FILTERS]))
                    conv_body = tf.layers.conv1d(
                        body_embeddings, var.NUM_FILTERS, filter_size)
                    relu_body = tf.nn.relu(tf.nn.bias_add(conv_body, b_head))
                    pool_body = tf.layers.max_pooling1d(
                        relu_body, var.CNN_BODY_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_body)

                cnn_out_vector = tf.reshape(
                    tf.concat(pooled_outputs, 2), [-1, var.NUM_FILTERS * len(var.FILTER_SIZES) * 2])
                cnn_out_vector = tf.nn.dropout(cnn_out_vector, keep_prob_pl)
                hidden_layer = tf.layers.dense(cnn_out_vector, var.HIDDEN_SIZE)

            else:
                hidden_layer = tf.nn.dropout(
                    tf.nn.relu(
                        tf.layers.dense(
                            features_pl,
                            var.HIDDEN_SIZE)),
                    keep_prob=keep_prob_pl)

            ### Label Prediction ###

            # Fully connected hidden layer with size based on LABEL_HIDDEN_SIZE
            # with original features concated
            if var.ADD_FEATURES_TO_LABEL_PRED:
                hidden_layer_p = tf.concat(
                    [p_features_pl, tf.identity(hidden_layer)], axis=1)
            else:
                hidden_layer_p = hidden_layer

            if var.LABEL_HIDDEN_SIZE is not None:
                hidden_layer_p = tf.nn.dropout(
                    tf.nn.relu(
                        tf.contrib.layers.linear(
                            hidden_layer_p,
                            var.LABEL_HIDDEN_SIZE)),
                    keep_prob=keep_prob_pl)

            logits_flat = tf.nn.dropout(
                tf.contrib.layers.linear(
                    hidden_layer_p,
                    var.TARGET_SIZE),
                keep_prob=keep_prob_pl)
            logits = tf.reshape(logits_flat, [batch_size, var.TARGET_SIZE])

            # Label loss and prediction
            p_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=stances_pl))
            softmaxed_logits = tf.nn.softmax(logits)
            predict = tf.argmax(softmaxed_logits, axis=1)

            ### Domain Prediction ###

            if var.USE_DOMAINS:
                # Gradient reversal layer
                hidden_layer_d = flip_gradient(hidden_layer, gr_pl)

                # Hidden layer size based on DOMAIN_HIDDEN_SIZE
                if var.DOMAIN_HIDDEN_SIZE is None:
                    domain_layer = hidden_layer_d
                else:
                    domain_layer = tf.nn.dropout(
                        tf.nn.relu(
                            tf.contrib.layers.linear(
                                hidden_layer_d,
                                DOMAIN_HIDDEN_SIZE)),
                        keep_prob=keep_prob_pl)

                d_logits_flat = tf.nn.dropout(
                    tf.contrib.layers.linear(
                        domain_layer,
                        var.DOMAIN_TARGET_SIZE),
                    keep_prob=keep_prob_pl)
                d_logits = tf.reshape(
                    d_logits_flat, [
                        batch_size, var.DOMAIN_TARGET_SIZE])

                # Domain loss and prediction
                d_loss = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=d_logits, labels=domains_pl))
                softmaxed_d_logits = tf.nn.softmax(d_logits)
                d_predict = tf.argmax(softmaxed_d_logits, axis=1)

            else:
                d_loss = tf.constant(0.0)
                d_predict = tf.constant(0.0)

            ### Regularization ###
            # L2 loss
            tf_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                for v in tf_vars if 'bias' not in v.name]) * var.L2_ALPHA

            # Define optimiser
            opt_func = tf.train.AdamOptimizer(lr_pl)
            grads, _ = tf.clip_by_global_norm(tf.gradients(
                var.RATIO_LOSS * p_loss + (1 - var.RATIO_LOSS) * d_loss + l2_loss, tf_vars), var.CLIP_RATIO)
            opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

            # Intialize saver to save model
            saver = tf.train.Saver()

            ### Training Model ###

            print("Training Model...")
            f.write("Training Model...\n")

            with tf.Session() as sess:
                # Load Pretrained Model if applicable
                if var.PRETRAINED_MODEL_PATH is not None:
                    print("Loading Saved Model...")
                    f.write("Loading Saved Model...\n")
                    saver.restore(sess, var.PRETRAINED_MODEL_PATH)

                else:
                    sess.run(tf.global_variables_initializer())

                for epoch in range(var.EPOCH_START, var.EPOCH_START + var.EPOCHS):
                    print("\n  EPOCH", epoch)
                    f.write("\n  EPOCH " + str(epoch))

                    # Adaption Parameter and Learning Rate
                    p = float(epoch) / var.TOTAL_EPOCHS
                    gr = 2. / (1. + np.exp(-10. * p)) - 1
                    lr = var.LR_FACTOR / (1. + 10 * p)**0.75

                    train_loss, train_p_loss, train_d_loss, train_l2_loss = 0, 0, 0, 0
                    train_l_pred, train_d_pred = [], []

                    # Organize and randomize order of training data from each
                    # dataset
                    index = 0
                    fnc_indices, snli_indices, fever_indices = [], [], []

                    if var.USE_FNC_DATA:
                        fnc_indices = list(
                            range(index, index + train_sizes['fnc']))
                        index += train_sizes['fnc']
                        var.RAND1.shuffle(fnc_indices)
                        for i in fnc_indices:
                            assert train_domains[i] == 0

                    if var.USE_FEVER_DATA:
                        fever_indices = list(
                            range(index, index + train_sizes['fever']))
                        index += train_sizes['fever']
                        var.RAND1.shuffle(fever_indices)
                        for i in fever_indices:
                            assert train_domains[i] == 2

                    if var.USE_SNLI_DATA:
                        snli_indices = list(
                            range(index, index + train_sizes['snli']))
                        index += train_sizes['snli']
                        var.RAND1.shuffle(snli_indices)
                        for i in snli_indices:
                            assert train_domains[i] == 1

                    # Use equal numbers of FNC and other data per epoch
                    if var.EXTRA_SAMPLES_PER_EPOCH is not None:

                        # Use equal numbers of agree/disagree labels per epoch
                        if var.BALANCE_LABELS:
                            fnc_agree_indices = [
                                i for i in fnc_indices if train_labels[i] == 0]
                            fnc_disagree_indices = [
                                i for i in fnc_indices if train_labels[i] == 1]

                            snli_agree_indices = [
                                i for i in snli_indices if train_labels[i] == 0]
                            snli_disagree_indices = [
                                i for i in snli_indices if train_labels[i] == 1]

                            fever_agree_indices = [
                                i for i in fever_indices if train_labels[i] == 0]
                            fever_disagree_indices = [
                                i for i in fever_indices if train_labels[i] == 1]

                            LABEL_SIZE = float('inf')
                            if var.USE_FNC_DATA:
                                LABEL_SIZE = min(
                                    LABEL_SIZE, len(fnc_agree_indices), len(fnc_disagree_indices))
                            if var.USE_SNLI_DATA:
                                LABEL_SIZE = min(
                                    LABEL_SIZE, len(snli_agree_indices), len(snli_disagree_indices))
                            if var.USE_FEVER_DATA:
                                LABEL_SIZE = min(
                                    LABEL_SIZE, len(fever_agree_indices), len(fever_disagree_indices))
                            train_indices = fnc_agree_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                fnc_disagree_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                snli_agree_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                snli_disagree_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                fever_agree_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                fever_disagree_indices[:LABEL_SIZE *
                                                       var.EXTRA_SAMPLES_PER_EPOCH]

                        # Don't balance labels but maintain same amount of data
                        # from each dataset
                        else:
                            LABEL_SIZE = float('inf')
                            if var.USE_FNC_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(fnc_indices))
                            if var.USE_FEVER_DATA:
                                LABEL_SIZE = min(
                                    LABEL_SIZE, len(fever_indices))
                            if var.USE_SNLI_DATA:
                                LABEL_SIZE = min(LABEL_SIZE, len(snli_indices))
                            train_indices = fnc_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                snli_indices[:LABEL_SIZE * var.EXTRA_SAMPLES_PER_EPOCH] + \
                                fever_indices[:LABEL_SIZE *
                                              var.EXTRA_SAMPLES_PER_EPOCH]

                    # Use all training data each epoch
                    else:
                        train_indices = fnc_indices + snli_indices + fever_indices

                    # Randomize order of training data
                    var.RAND2.shuffle(train_indices)

                    # Training epoch loop
                    for i in range(len(train_indices) // var.BATCH_SIZE + 1):

                        # Get training batches
                        batch_indices = train_indices[i *
                                                      var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
                        if len(batch_indices) == 0:
                            break
                        batch_stances = [train_labels[i]
                                         for i in batch_indices]
                        batch_domains = [train_domains[i]
                                         for i in batch_indices]

                        batch_feed_dict = {stances_pl: batch_stances,
                                           keep_prob_pl: var.TRAIN_KEEP_PROB,
                                           lr_pl: lr}

                        if var.USE_DOMAINS:
                            batch_feed_dict[domains_pl] = batch_domains
                            batch_feed_dict[gr_pl] = gr

                        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
                            batch_features = [train_tf_vectors[i]
                                              for i in batch_indices]
                            if var.USE_TF_VECTORS:
                                batch_feed_dict[features_pl] = batch_features
                            if var.ADD_FEATURES_TO_LABEL_PRED:
                                batch_feed_dict[p_features_pl] = batch_features

                        if var.USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relation_vectors = [
                                train_relation_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_relation_vectors

                        if var.USE_AVG_EMBEDDINGS:
                            batch_avg_embed_vectors = [
                                train_avg_embed_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_avg_embed_vectors

                        if var.USE_CNN_FEATURES:
                            batch_x_headlines = [
                                x_train_headlines[i] for i in batch_indices]
                            batch_x_bodies = [x_train_bodies[i]
                                              for i in batch_indices]
                            batch_feed_dict[headline_words_pl] = batch_x_headlines
                            batch_feed_dict[body_words_pl] = batch_x_bodies
                            batch_feed_dict[embedding_matrix_pl] = embedding_matrix

                        # Loss and predictions for each batch
                        _, lpred, dpred, ploss, dloss, l2loss = \
                            sess.run([opt_op,
                                      predict,
                                      d_predict,
                                      p_loss,
                                      d_loss,
                                      l2_loss],
                                     feed_dict=dict(batch_feed_dict))

                        # Total loss for current epoch
                        train_p_loss += ploss
                        train_d_loss += dloss
                        train_l2_loss += l2loss
                        train_loss += ploss + dloss + l2loss
                        train_l_pred.extend(lpred)

                        if var.USE_DOMAINS:
                            train_d_pred.extend(dpred)

                    # Record loss and accuracy information for train
                    actual_train_labels = [train_labels[i]
                                           for i in train_indices]
                    actual_train_domains = [
                        train_domains[i] for i in train_indices]
                    print_model_results(f, "Train", train_l_pred, actual_train_labels, train_d_pred, actual_train_domains,
                                        train_p_loss, train_d_loss, train_l2_loss, var.USE_DOMAINS)

                    # Record loss and accuracy for val
                    if var.VALIDATION_SET_SIZE is not None and var.VALIDATION_SET_SIZE > 0:
                        val_indices = list(range(SIZE_VAL))
                        val_loss, val_p_loss, val_d_loss, val_l2_loss = 0, 0, 0, 0
                        val_l_pred, val_d_pred = [], []

                        for i in range(int(SIZE_VAL) // var.BATCH_SIZE + 1):
                            batch_indices = val_indices[i *
                                                        var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
                            if len(batch_indices) == 0:
                                break
                            batch_stances = [val_labels[i]
                                             for i in batch_indices]
                            batch_domains = [val_domains[i]
                                             for i in batch_indices]

                            batch_feed_dict = {stances_pl: batch_stances,
                                               keep_prob_pl: 1.0}

                            if var.USE_DOMAINS:
                                batch_feed_dict[gr_pl] = 1.0
                                batch_feed_dict[domains_pl] = batch_domains

                            if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
                                batch_features = [val_tf_vectors[i]
                                                  for i in batch_indices]
                                if var.USE_TF_VECTORS:
                                    batch_feed_dict[features_pl] = batch_features
                                if var.ADD_FEATURES_TO_LABEL_PRED:
                                    batch_feed_dict[p_features_pl] = batch_features

                            if var.USE_RELATIONAL_FEATURE_VECTORS:
                                batch_relation_vectors = [
                                    val_relation_vectors[i] for i in batch_indices]
                                batch_feed_dict[features_pl] = batch_relation_vectors

                            if var.USE_AVG_EMBEDDINGS:
                                batch_avg_embed_vectors = [
                                    val_avg_embed_vectors[i] for i in batch_indices]
                                batch_feed_dict[features_pl] = batch_avg_embed_vectors

                            if var.USE_CNN_FEATURES:
                                batch_x_headlines = [
                                    x_val_headlines[i] for i in batch_indices]
                                batch_x_bodies = [x_val_bodies[i]
                                                  for i in batch_indices]
                                batch_feed_dict[headline_words_pl] = batch_x_headlines
                                batch_feed_dict[body_words_pl] = batch_x_bodies
                                batch_feed_dict[embedding_matrix_pl] = embedding_matrix

                            # Record loss and accuracy information for test
                            lpred, dpred, ploss, dloss, l2loss = \
                                sess.run(
                                    [predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)

                            # Total loss for current epoch
                            val_p_loss += ploss
                            val_d_loss += dloss
                            val_l2_loss += l2loss
                            val_loss += ploss + dloss + l2loss
                            val_l_pred.extend(lpred)
                            if var.USE_DOMAINS:
                                val_d_pred.extend(dpred)

                        print_model_results(f, "Val", val_l_pred, val_labels, val_d_pred, val_domains,
                                            val_p_loss, val_d_loss, val_l2_loss, var.USE_DOMAINS)

                        # Save best test label loss model
                        if val_p_loss < best_loss:
                            best_loss = val_p_loss
                            saver.save(sess, SAVE_MODEL_PATH)
                            print("\n    New Best Val Loss")
                            f.write("\n    New Best Val Loss\n")

                    test_indices = list(range(SIZE_TEST))
                    test_loss, test_p_loss, test_d_loss, test_l2_loss = 0, 0, 0, 0
                    test_l_pred, test_d_pred = [], []

                    for i in range(SIZE_TEST // var.BATCH_SIZE + 1):

                        batch_indices = test_indices[i *
                                                     var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
                        if len(batch_indices) == 0:
                            break
                        batch_stances = [test_labels[i] for i in batch_indices]
                        batch_domains = [test_domains[i]
                                         for i in batch_indices]

                        batch_feed_dict = {stances_pl: batch_stances,
                                           keep_prob_pl: 1.0}

                        if var.USE_DOMAINS:
                            batch_feed_dict[gr_pl] = 1.0
                            batch_feed_dict[domains_pl] = batch_domains

                        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
                            batch_features = [test_tf_vectors[i]
                                              for i in batch_indices]
                            if var.USE_TF_VECTORS:
                                batch_feed_dict[features_pl] = batch_features
                            if var.ADD_FEATURES_TO_LABEL_PRED:
                                batch_feed_dict[p_features_pl] = batch_features

                        if var.USE_RELATIONAL_FEATURE_VECTORS:
                            batch_relation_vectors = [
                                test_relation_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_relation_vectors

                        if var.USE_AVG_EMBEDDINGS:
                            batch_avg_embed_vectors = [
                                test_avg_embed_vectors[i] for i in batch_indices]
                            batch_feed_dict[features_pl] = batch_avg_embed_vectors

                        if var.USE_CNN_FEATURES:
                            batch_x_headlines = [
                                x_test_headlines[i] for i in batch_indices]
                            batch_x_bodies = [x_test_bodies[i]
                                              for i in batch_indices]
                            batch_feed_dict[headline_words_pl] = batch_x_headlines
                            batch_feed_dict[body_words_pl] = batch_x_bodies
                            batch_feed_dict[embedding_matrix_pl] = embedding_matrix

                        # Record loss and accuracy information for test
                        lpred, dpred, ploss, dloss, l2loss = \
                            sess.run([predict, d_predict, p_loss, d_loss,
                                      l2_loss], feed_dict=batch_feed_dict)

                        # Total loss for current epoch
                        test_p_loss += ploss
                        test_d_loss += dloss
                        test_l2_loss += l2loss
                        test_loss += ploss + dloss + l2loss
                        test_l_pred.extend(lpred)
                        if var.USE_DOMAINS:
                            test_d_pred.extend(dpred)

                    # Print and Write test results
                    print_model_results(f, "Test", test_l_pred, test_labels, test_d_pred, test_domains,
                                        test_p_loss, test_d_loss, test_l2_loss, var.USE_DOMAINS)

                #saver.restore(sess, SAVE_MODEL_PATH)
                #test_l_pred = sess.run([predict], feed_dict = test_feed_dict)
                #save_predictions(test_l_pred, test_labels, PREDICTION_FILE)


if __name__ == "__main__":
    process_data()
    #train_model()
