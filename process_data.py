import numpy as np
import os
import var

from shutil import copyfile
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from util import get_fnc_data, get_snli_data, get_fever_data, \
    get_vectorizers, get_feature_vectors, get_relational_feature_vectors, \
    remove_stop_words, get_average_embeddings, remove_data_with_label


def process_data():
    '''
    Given the parameters in var.py, extract training, validation, and test
    data and save them into pickle files to be used by train_model.py. The
    pickle data will be saved into PICKLE_SAVE_FOLDER.

    In order to extract data, the following must be stored in the data
    folder with the appropriate path set in var.py
      - trained Word2Vec embeddings
      - fnc data
      - snli data
      - fever data
    '''

    if not os.path.exists(var.PICKLE_SAVE_FOLDER):
        raise Exception("Specified PICKLE_SAVE_FOLDER doesn't exist")

    with open(var.PICKLE_LOG_FILE, 'w') as f:
        # Save parameters
        copyfile('var.py', os.path.join(var.PICKLE_SAVE_FOLDER, 'var.py'))

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
            f.write("EXTRACTING FNC DATA\n")
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

            train_body_ids = set(unique_body_ids[i]
                                 for i in indices[:FNC_TRAIN_END])
            val_body_ids = set(unique_body_ids[i]
                               for i in indices[FNC_TRAIN_END:])

            train_indices = set(
                i for i in range(
                    len(fnc_body_ids_train)) if fnc_body_ids_train[i] in train_body_ids)
            val_indices = set(
                i for i in range(
                    len(fnc_body_ids_train)) if fnc_body_ids_train[i] in val_body_ids)

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
            f.write("EXTRACTING FEVER DATA\n")
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

            train_claims = set([claim_list[i]
                                for i in claim_indices[:FEVER_TRAIN_END]])
            val_claims = set([claim_list[i]
                              for i in claim_indices[FEVER_TRAIN_END:FEVER_VAL_END]])
            test_claims = set([claim_list[i]
                               for i in claim_indices[FEVER_VAL_END:]])

            train_indices = [
                i for i in range(
                    len(fever_headlines)) if fever_headlines[i] in train_claims]
            val_indices = [
                i for i in range(
                    len(fever_headlines)) if fever_headlines[i] in val_claims]
            test_indices = [
                i for i in range(
                    len(fever_headlines)) if fever_headlines[i] in test_claims]

            # Add training and val data into overall training and val lists
            train_headlines += [fever_headlines[i] for i in train_indices]
            train_bodies += [fever_bodies[i] for i in train_indices]
            train_labels += [fever_labels[i] for i in train_indices]
            train_domains += [fever_domains[i] for i in train_indices]

            val_headlines += [fever_headlines[i]
                              for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [fever_bodies[i] for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [fever_labels[i] for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [fever_domains[i]
                            for i in val_indices][:VAL_SIZE_CAP]

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
            f.write("EXTRACTING SNLI DATA\n")
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
            val_s2 = set(s2_list[i]
                         for i in s2_indices[SNLI_TRAIN_END:SNLI_VAL_END])
            test_s2 = set(s2_list[i] for i in s2_indices[SNLI_VAL_END:])

            train_indices = [
                i for i in range(
                    len(snli_s2_train)) if snli_s2_train[i] in train_s2]
            val_indices = [
                i for i in range(
                    len(snli_s2_train)) if snli_s2_train[i] in val_s2]
            test_indices = [
                i for i in range(
                    len(snli_s2_train)) if snli_s2_train[i] in test_s2]

            # Add the train and val data to the overall train and val
            # lists
            train_headlines += [snli_s1_train[i] for i in train_indices]
            train_bodies += [snli_s2_train[i] for i in train_indices]
            train_labels += [snli_labels_train[i] for i in train_indices]
            train_domains += [snli_domains[i] for i in train_indices]

            val_headlines += [snli_s1_train[i]
                              for i in val_indices][:VAL_SIZE_CAP]
            val_bodies += [snli_s2_train[i]
                           for i in val_indices][:VAL_SIZE_CAP]
            val_labels += [snli_labels_train[i]
                           for i in val_indices][:VAL_SIZE_CAP]
            val_domains += [snli_domains[i]
                            for i in val_indices][:VAL_SIZE_CAP]

            snli_headlines_test = [snli_s1_train[i] for i in test_indices]
            snli_bodies_test = [snli_s2_train[i] for i in test_indices]
            snli_labels_test = [snli_labels_train[i] for i in test_indices]
            snli_domains_test = [snli_domains[i] for i in train_indices]

            train_sizes['snli'] = len(train_indices)
            val_sizes['snli'] = len(val_indices)

            if VAL_SIZE_CAP is None:
                VAL_SIZE_CAP = len(val_headlines)

        # Select what test data to use
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

        if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
            print("Creating Vectorizers...")
            f.write("Creating Vectorizers...\n")

            vec_train_data = train_headlines + train_bodies
            vec_test_data = val_headlines + val_bodies + test_headlines + test_bodies

            # Only train TF & TFIDF vectorizers with FNC data
            if var.ONLY_VECT_FNC and var.USE_FNC_DATA:
                vec_train_data = fnc_headlines_train + fnc_bodies_train
                vec_test_data = fnc_headlines_test + fnc_bodies_test 

            bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = get_vectorizers(
                vec_train_data, vec_test_data, var.MAX_FEATURES)
            del vec_train_data
            del vec_test_data

            print("Getting Feature Vectors...")
            f.write("Getting Feature Vectors...\n")

            print("  train...")
            f.write("  train...\n")
            train_tf_vectors = get_feature_vectors(
                train_headlines,
                train_bodies,
                bow_vectorizer,
                tfreq_vectorizer,
                tfidf_vectorizer)
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "train_tf_vectors.npy",
                train_tf_vectors)
            f.write("    Number of Feature Vectors: " +
                    str(len(train_tf_vectors)) + "\n")

            print("  val...")
            f.write("  val...\n")
            val_tf_vectors = get_feature_vectors(
                val_headlines,
                val_bodies,
                bow_vectorizer,
                tfreq_vectorizer,
                tfidf_vectorizer)
            np.save(
                var.PICKLE_SAVE_FOLDER + "val_tf_vectors.npy",
                val_tf_vectors)
            f.write("    Number of Feature Vectors: " +
                    str(len(val_tf_vectors)) + "\n")

            print("  test...")
            f.write("  test...\n")
            test_tf_vectors = get_feature_vectors(
                test_headlines,
                test_bodies,
                bow_vectorizer,
                tfreq_vectorizer,
                tfidf_vectorizer)

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
            train_avg_embed_vectors = [
                np.concatenate(
                    [
                        train_headline_avg_embeddings[i],
                        train_body_avg_embeddings[i]]) for i in range(
                    len(train_headline_avg_embeddings))]
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "train_avg_embed_vectors.npy",
                train_avg_embed_vectors)
            del train_avg_embed_vectors

            print("  Combining Val Vectors...")
            f.write("  Combining Val Vectors...")
            val_avg_embed_vectors = [
                np.concatenate(
                    [
                        val_headline_avg_embeddings[i],
                        train_body_avg_embeddings[i]]) for i in range(
                    len(val_headline_avg_embeddings))]
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "val_avg_embed_vectors.npy",
                val_avg_embed_vectors)
            del val_avg_embed_vectors

            print("  Combining Test Vectors...")
            f.write("  Combining Test Vectors...\n")
            test_avg_embed_vectors = [
                np.concatenate(
                    [
                        test_headline_avg_embeddings[i],
                        test_body_avg_embeddings[i]]) for i in range(
                    len(test_headline_avg_embeddings))]
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
            x_val_bodies = pad_sequences(
                val_body_seq, maxlen=var.CNN_BODY_LENGTH)

            x_test_headlines = pad_sequences(
                test_headline_seq, maxlen=var.CNN_HEADLINE_LENGTH)
            x_test_bodies = pad_sequences(
                test_body_seq, maxlen=var.CNN_BODY_LENGTH)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_train_headlines.npy",
                x_train_headlines)
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_train_bodies.npy",
                x_train_bodies)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_val_headlines.npy",
                x_val_headlines)
            np.save(var.PICKLE_SAVE_FOLDER + "x_val_bodies.npy", x_val_bodies)

            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_test_headlines.npy",
                x_test_headlines)
            np.save(
                var.PICKLE_SAVE_FOLDER +
                "x_test_bodies.npy",
                x_test_bodies)

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


if __name__ == "__main__":
    process_data()
