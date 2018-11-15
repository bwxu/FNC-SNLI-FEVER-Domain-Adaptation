import numpy as np
import tensorflow as tf
import var

from util import print_model_results, get_prediction_accuracies, get_composite_score, get_f1_scores, save_predictions

def test_model():

    print("Loading val vectors...")
    val_labels = np.load(var.PICKLE_SAVE_FOLDER + "val_labels.npy")
    val_domains = np.load(var.PICKLE_SAVE_FOLDER + "val_domains.npy")

    print("Loading test vectors...")
    test_labels = np.load(var.PICKLE_SAVE_FOLDER + "test_labels.npy")
    test_domains = np.load(var.PICKLE_SAVE_FOLDER + "test_domains.npy")

    # TF vectors always need to be loaded for related, unrelated model
    print("Loading TF vectors...")
    val_tf_vectors = np.load(
        var.PICKLE_SAVE_FOLDER + "val_tf_vectors.npy")
    test_tf_vectors = np.load(
        var.PICKLE_SAVE_FOLDER + "test_tf_vectors.npy")

    SIZE_VAL = len(val_tf_vectors)
    SIZE_TEST = len(test_tf_vectors)

    if var.USE_RELATIONAL_FEATURE_VECTORS:
        print("Loading relation vectors...")

        val_relation_vectors = np.load(
            var.PICKLE_SAVE_FOLDER + "val_relation_vectors.npy")
        test_relation_vectors = np.load(
            var.PICKLE_SAVE_FOLDER + "test_relation_vectors.npy")

        SIZE_VAL = len(val_relation_vectors)
        SIZE_TEST = len(test_relation_vectors)

    if var.USE_AVG_EMBEDDINGS:
        print("Loading avg embedding vectors...")

        val_avg_embed_vectors = np.load(
            var.PICKLE_SAVE_FOLDER + "val_avg_embed_vectors.npy")
        test_avg_embed_vectors = np.load(
            var.PICKLE_SAVE_FOLDER + "test_avg_embed_vectors.npy")

        SIZE_VAL = len(val_avg_embed_vectors)
        SIZE_TEST = len(test_avg_embed_vectors)

    if var.USE_CNN_FEATURES:
        print("Loading CNN vectors...")

        x_val_headlines = np.load(
            var.PICKLE_SAVE_FOLDER + "x_val_headlines.npy")
        x_val_bodies = np.load(
            var.PICKLE_SAVE_FOLDER + "x_val_bodies.npy")
        x_test_headlines = np.load(
            var.PICKLE_SAVE_FOLDER + "x_test_headlines.npy")
        x_test_bodies = np.load(
            var.PICKLE_SAVE_FOLDER + "x_test_bodies.npy")

        SIZE_VAL = len(x_val_headlines)
        SIZE_TEST = len(x_test_headlines)

        embedding_matrix = np.load(
            var.PICKLE_SAVE_FOLDER + "embedding_matrix.npy")
        word_index = np.load(
            var.PICKLE_SAVE_FOLDER + "word_index.npy")
        word_index = word_index.item()

    graph1 = tf.Graph()
    graph2 = tf.Graph()

    var.USE_UNRELATED_LABEL = True

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
        d_loss = graph.get_tensor_by_name("d_loss:0")
        d_predict = graph.get_tensor_by_name("d_predict:0")
        l2_loss = graph.get_tensor_by_name("l2_loss:0")

        # Record loss and accuracies for ru, assumed to be the original Tf model
        test_indices = list(range(SIZE_TEST))
        ru_test_loss, ru_test_p_loss, ru_test_d_loss, ru_test_l2_loss = 0, 0, 0, 0
        ru_test_l_pred, ru_test_d_pred = [], []

        freq = [0, 0, 0, 0]
        for label in test_labels:
            freq[label] += 1
        print(freq)

        # change to 0, 3 test labels for ru
        ru_test_labels = [0 if label in [0, 1, 2] else 3 for label in test_labels]

        freq = [0, 0, 0, 0]
        for label in ru_test_labels:
            freq[label] += 1
        print(freq)

        for i in range(SIZE_TEST // var.BATCH_SIZE + 1):

            batch_indices = test_indices[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
            if len(batch_indices) == 0:
                break
            batch_stances = [ru_test_labels[i] for i in batch_indices]
            batch_domains = [test_domains[i]
                             for i in batch_indices]

            batch_feed_dict = {stances_pl: batch_stances,
                               keep_prob_pl: 1.0}
       
            batch_features = [test_tf_vectors[i]
                              for i in batch_indices]
            batch_feed_dict[features_pl] = batch_features
               
            # Record loss and accuracy information for test
            lpred, dpred, ploss, dloss, l2loss = \
                sess.run([p_predict, d_predict, p_loss, d_loss,
                          l2_loss], feed_dict=batch_feed_dict)

            # Total loss for current epoch
            ru_test_p_loss += ploss
            ru_test_d_loss += dloss
            ru_test_l2_loss += l2loss
            ru_test_loss += ploss + dloss + l2loss
            ru_test_l_pred.extend(lpred)

        # Print and Write test results
        with open(var.TEST_RESULTS_FILE, 'a') as f:
            print_model_results(
                f,
                "Test Related Unrelated",
                ru_test_l_pred,
                ru_test_labels,
                ru_test_d_pred,
                test_domains,
                ru_test_p_loss,
                ru_test_d_loss,
                ru_test_l2_loss,
                False)
            print()

    var.USE_UNRELATED_LABEL = False
    var.USE_DISCUSS_LABEL = True

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
        d_loss = graph.get_tensor_by_name("d_loss:0")
        d_predict = graph.get_tensor_by_name("d_predict:0")
        l2_loss = graph.get_tensor_by_name("l2_loss:0")

        test_indices = list(range(SIZE_TEST))
        three_test_loss, three_test_p_loss, three_test_d_loss, three_test_l2_loss = 0, 0, 0, 0
        three_test_l_pred, three_test_d_pred = [], []

        for i in range(SIZE_TEST // var.BATCH_SIZE + 1):

            batch_indices = test_indices[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
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
                sess.run([p_predict, d_predict, p_loss, d_loss,
                          l2_loss], feed_dict=batch_feed_dict)

            # Total loss for current epoch
            three_test_p_loss += ploss
            three_test_d_loss += dloss
            three_test_l2_loss += l2loss
            three_test_loss += ploss + dloss + l2loss
            three_test_l_pred.extend(lpred)
            if var.USE_DOMAINS:
                three_test_d_pred.extend(dpred)

        # Print and Write test results
        with open(var.TEST_RESULTS_FILE, 'a') as f:
            print_model_results(
                f,
                "Test 3 Label",
                three_test_l_pred,
                test_labels,
                three_test_d_pred,
                test_domains,
                three_test_p_loss,
                three_test_d_loss,
                three_test_l2_loss,
                var.USE_DOMAINS)
            print()

    var.USE_UNRELATED_LABEL = True
    var.USE_DISCUSS_LABEL = True

    predictions = list(ru_test_l_pred)
    for i in range(len(predictions)):
        if predictions[i] == 0:
            predictions[i] = three_test_l_pred[i]
    print("composite score: ", get_composite_score(predictions, test_labels))
    print("label accuracies: ", get_prediction_accuracies(predictions, test_labels, 4))
    print("f1 scores: ", get_f1_scores(predictions, test_labels, 4))
    print()

    save_predictions(predictions, var.TEST_RESULTS_CSV)


if __name__ == "__main__":
    test_model()
