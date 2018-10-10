import numpy as np
import tensorflow as tf
import var
from csv import DictReader, DictWriter

from util import print_model_results, save_predictions

def test_model():

    print("Loading val vectors...")
    val_labels = np.load(var.PICKLE_SAVE_FOLDER + "val_labels.npy")
    val_domains = np.load(var.PICKLE_SAVE_FOLDER + "val_domains.npy")

    print("Loading test vectors...")
    test_labels = np.load(var.PICKLE_SAVE_FOLDER + "test_labels.npy")
    test_domains = np.load(var.PICKLE_SAVE_FOLDER + "test_domains.npy")

    if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
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

    with tf.Session() as sess:
        print(var.SAVE_MODEL_PATH)
        print(var.SAVE_FOLDER)
        saver = tf.train.import_meta_graph(var.SAVE_MODEL_PATH + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(var.SAVE_FOLDER))

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
                [p_predict, d_predict, p_loss, d_loss, l2_loss], feed_dict=batch_feed_dict)

            # Total loss for current epoch
            val_p_loss += ploss
            val_d_loss += dloss
            val_l2_loss += l2loss
            val_loss += ploss + dloss + l2loss
            val_l_pred.extend(lpred)
            if var.USE_DOMAINS:
                val_d_pred.extend(dpred)

        with open(var.TEST_RESULTS_FILE, 'w') as f:
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
                sess.run([p_predict, d_predict, p_loss, d_loss,
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
        with open(var.TEST_RESULTS_FILE, 'a') as f:
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
            print()
        
        save_predictions(test_l_pred, var.TEST_RESULTS_CSV)

if __name__ == "__main__":
    test_model()
