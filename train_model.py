import tensorflow as tf
import numpy as np
import os
import var

from shutil import copyfile
from flip_gradient import flip_gradient
from util import save_predictions, print_model_results, get_label_freq


def train_model():
    '''
    Given the parameters in var.py, train the specified model with 
    the specified data. Save into var.SAVE_FOLDER the checkpoint
    with the lowest val loss along with a copy of var.py and a 
    training log
    '''

    if not os.path.exists(var.SAVE_FOLDER):
        raise Exception("Specified SAVE_FOLDER doesn't exist")

    with open(var.TRAINING_LOG_FILE, 'w') as f:
        copyfile('var.py', os.path.join(var.SAVE_FOLDER, 'var.py'))

        ### LOAD DATA ###

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

        freq_dict = get_label_freq(test_labels)
        print("TEST LABEL FREQ = ", freq_dict)
        f.write("TEST LABEL FREQ = [ ")
        for label in freq_dict:
            f.write("%d: %d, " % (label, freq_dict[label]))
        f.write("]\n")

        ### DEFINE MODEL ###

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
                FEATURE_VECTOR_SIZE = var.NUM_FILTERS * \
                    len(var.FILTER_SIZES) * 2

            stances_pl = tf.placeholder(tf.int64, [None], name="stances_pl")
            keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob_pl")
            domains_pl = tf.placeholder(tf.int64, [None], name="domains_pl")
            gr_pl = tf.placeholder(tf.float32, [], name="gr_pl")
            lr_pl = tf.placeholder(tf.float32, [], name="lr_pl")

            # If the model uses tf vectors, relational vectors, or average
            # embeddings feed the features to a hidden layer 
            if var.USE_TF_VECTORS or var.USE_RELATIONAL_FEATURE_VECTORS or var.USE_AVG_EMBEDDINGS:
                features_pl = tf.placeholder(
                    tf.float32, [
                        None, FEATURE_VECTOR_SIZE], name="features_pl")
                batch_size = tf.shape(features_pl)[0]

                hidden_layer = tf.nn.dropout(
                    tf.nn.relu(
                        tf.layers.dense(
                            features_pl,
                            var.HIDDEN_SIZE)),
                    keep_prob=keep_prob_pl)

            # If inputs for a CNN model are given, create a CNN model that
            # convolves the headline and body inputs
            elif var.USE_CNN_FEATURES:
                embedding_matrix_pl = tf.placeholder(
                    tf.float32, [len(word_index) + 1, var.EMBEDDING_DIM], name="embedding_matrix_pl")
                W = tf.Variable(
                    tf.constant(
                        0.0,
                        shape=[
                            len(word_index) + 1,
                            var.EMBEDDING_DIM]),
                    trainable=False)
                embedding_init = W.assign(embedding_matrix_pl)
                headline_words_pl = tf.placeholder(
                    tf.int64, [None, len(x_train_headlines[0])], name="headline_words_pl")
                body_words_pl = tf.placeholder(
                    tf.int64, [None, len(x_train_bodies[0])], name="body_words_pl")
                batch_size = tf.shape(headline_words_pl)[0]

                pooled_outputs = []

                headline_embeddings = tf.nn.embedding_lookup(
                    embedding_init, headline_words_pl)
                body_embeddings = tf.nn.embedding_lookup(
                    embedding_init, body_words_pl)
                
                # Convolve headline
                for filter_size in var.FILTER_SIZES:
                    b_head = tf.Variable(tf.constant(
                        0.1, shape=[var.NUM_FILTERS]))
                    conv_head = tf.layers.conv1d(
                        headline_embeddings, var.NUM_FILTERS, filter_size)
                    relu_head = tf.nn.relu(tf.nn.bias_add(conv_head, b_head))
                    pool_head = tf.layers.max_pooling1d(
                        relu_head, var.CNN_HEADLINE_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_head)

                # Convolve Body
                for filter_size in var.FILTER_SIZES:
                    b_body = tf.Variable(tf.constant(
                        0.1, shape=[var.NUM_FILTERS]))
                    conv_body = tf.layers.conv1d(
                        body_embeddings, var.NUM_FILTERS, filter_size)
                    relu_body = tf.nn.relu(tf.nn.bias_add(conv_body, b_head))
                    pool_body = tf.layers.max_pooling1d(
                        relu_body, var.CNN_BODY_LENGTH - filter_size + 1, 1)
                    pooled_outputs.append(pool_body)

                cnn_out_vector = tf.reshape(tf.concat(
                    pooled_outputs, 2), [-1, var.NUM_FILTERS * len(var.FILTER_SIZES) * 2])
                cnn_out_vector = tf.nn.dropout(cnn_out_vector, keep_prob_pl)
                hidden_layer = tf.layers.dense(cnn_out_vector, var.HIDDEN_SIZE)

            # If addiitonal features are to be added to the hidden layer,
            # declare the placeholder
            if var.ADD_FEATURES_TO_LABEL_PRED:
                p_features_pl = tf.placeholder(
                    tf.float32, [
                        None, len(
                            train_tf_vectors[0])], name="p_features_pl")

                hidden_layer_p = tf.concat(
                    [p_features_pl, tf.identity(hidden_layer)], axis=1)
            else:
                hidden_layer_p = hidden_layer

            # Add additional hidden layer if specified
            if var.LABEL_HIDDEN_SIZE is not None:
                hidden_layer_p = tf.nn.dropout(
                    tf.nn.relu(
                        tf.contrib.layers.linear(
                            hidden_layer_p,
                            var.LABEL_HIDDEN_SIZE)),
                    keep_prob=keep_prob_pl)

            # Feed into a final hidden layer to get logits
            logits_flat = tf.nn.dropout(
                tf.contrib.layers.linear(
                    hidden_layer_p,
                    var.TARGET_SIZE),
                keep_prob=keep_prob_pl)
            logits = tf.reshape(logits_flat, [batch_size, var.TARGET_SIZE])

            # Label loss and prediction
            p_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=stances_pl), name="p_loss")
            softmaxed_logits = tf.nn.softmax(logits)
            predict = tf.argmax(softmaxed_logits, axis=1, name="p_predict")

            # If domain adaptation is to be used, attach it after features
            # are extracted
            if var.USE_DOMAINS:
                # Gradient reversal layer
                hidden_layer_d = flip_gradient(hidden_layer, gr_pl)

                # Add a hidden layer to domain component if specified
                if var.DOMAIN_HIDDEN_SIZE is None:
                    domain_layer = hidden_layer_d
                else:
                    domain_layer = tf.nn.dropout(
                        tf.nn.relu(
                            tf.contrib.layers.linear(
                                hidden_layer_d,
                                var.DOMAIN_HIDDEN_SIZE)),
                        keep_prob=keep_prob_pl)

                # Last hidden layer to determine logits for domain pred
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
                        logits=d_logits, labels=domains_pl), name="d_loss")
                softmaxed_d_logits = tf.nn.softmax(d_logits)
                d_predict = tf.argmax(softmaxed_d_logits, axis=1, name="d_predict")

            # If domain adaptation is not used, set d_loss and d_predict
            # to 0 to indicate this.
            else:
                d_loss = tf.constant(0.0, name="d_loss")
                d_predict = tf.constant(0.0, name="d_predict")

            # Add L2 loss regularization to the model
            tf_vars = tf.trainable_variables()
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v)
                                for v in tf_vars if 'bias' not in v.name]), var.L2_ALPHA, name="l2_loss")

            # Define what optimizer to use to train the model
            opt_func = tf.train.AdamOptimizer(lr_pl)
            grads, _ = tf.clip_by_global_norm(tf.gradients(
                var.RATIO_LOSS * p_loss + (1 - var.RATIO_LOSS) * d_loss + l2_loss, tf_vars), var.CLIP_RATIO)
            opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

            # Intialize saver to save model checkpoints
            saver = tf.train.Saver()

            ### Model Training ###

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

                # record best val loss for current model to determine which models
                # to save.
                best_loss = float('Inf')
                
                for epoch in range(var.EPOCH_START,
                                   var.EPOCH_START + var.EPOCHS):
                    print("\n  EPOCH", epoch)
                    f.write("\n  EPOCH " + str(epoch))

                    # Adaption Parameter and Learning Rate
                    # https://arxiv.org/pdf/1409.7495.pdf
                    p = float(epoch) / var.TOTAL_EPOCHS

                    # Define gradient reversal to steadily increase
                    # influence of gr to allow classifier to train
                    # reasonably before application
                    gr = 2. / (1. + np.exp(-10. * p)) - 1
                    # Define learning rate to decay over time to 
                    # promote convergence
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

                    # Fix the ratio of FNC to data from other sources to be 
                    # var.EXTRA_SAMPLES_PER_EPOCH
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
                    print_model_results(
                        f,
                        "Train",
                        train_l_pred,
                        actual_train_labels,
                        train_d_pred,
                        actual_train_domains,
                        train_p_loss,
                        train_d_loss,
                        train_l2_loss,
                        var.USE_DOMAINS)

                    # Record loss and accuracy for val
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
                        lpred, dpred, ploss, dloss, l2loss = sess.run(
                            [predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)

                        # Total loss for current epoch
                        val_p_loss += ploss
                        val_d_loss += dloss
                        val_l2_loss += l2loss
                        val_loss += ploss + dloss + l2loss
                        val_l_pred.extend(lpred)
                        if var.USE_DOMAINS:
                            val_d_pred.extend(dpred)

                    print_model_results(
                        f,
                        "Val",
                        val_l_pred,
                        val_labels,
                        val_d_pred,
                        val_domains,
                        val_p_loss,
                        val_d_loss,
                        val_l2_loss,
                        var.USE_DOMAINS)

                    # Save best test label loss model
                    if val_p_loss < best_loss:
                        best_loss = val_p_loss
                        saver.save(sess, var.SAVE_MODEL_PATH + str(model_num))
                        print("\n    New Best Val Loss")
                        f.write("\n    New Best Val Loss\n")

                    # Record loss and accuracies for test
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
                    print_model_results(
                        f,
                        "Test",
                        test_l_pred,
                        test_labels,
                        test_d_pred,
                        test_domains,
                        test_p_loss,
                        test_d_loss,
                        test_l2_loss,
                        var.USE_DOMAINS)


if __name__ == "__main__":
    train_model()
