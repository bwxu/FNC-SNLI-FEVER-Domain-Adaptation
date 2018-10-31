# Data processing save parameters
PICKLE_FOLDER_NAME = "fever_older"
PICKLE_SAVE_FOLDER = "pickle_data/" + PICKLE_FOLDER_NAME + "/"
PICKLE_LOG_FILE = PICKLE_SAVE_FOLDER + "log.txt"

# Model save parameters
MODEL_NAME = "fever_tf_cnn_old"
DATE_CREATED = "oct_30_b"
SAVE_FOLDER = "models/" + DATE_CREATED + "/" + MODEL_NAME + "/"
SAVE_MODEL_PATH = SAVE_FOLDER + MODEL_NAME
TRAINING_LOG_FILE = SAVE_FOLDER + "training.txt"
TEST_RESULTS_FILE = SAVE_FOLDER + "test_results.txt"
TEST_RESULTS_CSV = SAVE_FOLDER + "test_predictions.csv"
UNLABELED_CSV = "competition_test_stances_unlabeled.csv"

# Checkpoints for 4 label fnc testing
RELATED_UNRELATED_MODEL_DATE = "oct_3"
RELATED_UNRELATED_MODEL_NAME = "fnc_related_unrelated_tf"
RELATED_UNRELATED_MODEL_FOLDER = "models/" + RELATED_UNRELATED_MODEL_DATE + "/" + RELATED_UNRELATED_MODEL_NAME + "/"
RELATED_UNRELATED_MODEL_PATH = RELATED_UNRELATED_MODEL_FOLDER + RELATED_UNRELATED_MODEL_NAME

THREE_LABEL_MODEL_DATE = "oct_3"
THREE_LABEL_MODEL_NAME = "fnc_fever_cnn_tf_dann_3_no_4"
THREE_LABEL_MODEL_FOLDER = "models/" + THREE_LABEL_MODEL_DATE + "/" + THREE_LABEL_MODEL_NAME + "/"
THREE_LABEL_MODEL_PATH = THREE_LABEL_MODEL_FOLDER + THREE_LABEL_MODEL_NAME

# Path to the ckpt of a saved model to load
PRETRAINED_MODEL_PATH = None

# Train/Val label options
USE_UNRELATED_LABEL = False
USE_DISCUSS_LABEL = False

# Select train and val datasets
USE_FNC_DATA = False
USE_SNLI_DATA = False
USE_FEVER_DATA = True
USE_FEVER_SENT = False

# Select test dataset
TEST_DATASET = "FEVER"
if TEST_DATASET not in ["FNC", "FEVER"]:
    raise Exception("TEST_DATASET must be FNC, FEVER")
if TEST_DATASET == "FNC" and not USE_FNC_DATA:
    raise Exception("Must use dataset to use test data")
if TEST_DATASET == "SNLI" and not USE_SNLI_DATA:
    raise Exception("Must use dataset to use test data")
if TEST_DATASET == "FEVER" and not USE_FEVER_DATA:
    raise Exception("Must use dataset to use test data")

# Only train TF vectorizer with FNC data
ONLY_VECT_FNC = True

# Use equal numbers of agree and disagree data
BALANCE_LABELS = True

# Only use fnc data for validation
ONLY_FNC_VAL = False

# Use Domain Adaptation
USE_DOMAINS = False

# One of these must be selected as the primary input for training
# but multiple may be selected when processing data
USE_TF_VECTORS = False
USE_RELATIONAL_FEATURE_VECTORS = False
USE_AVG_EMBEDDINGS = False
USE_CNN_FEATURES = True

# Adds TF vectors to features before label prediction
ADD_FEATURES_TO_LABEL_PRED = True

# Training params
EPOCHS = 20
TOTAL_EPOCHS = 30
SAVE_EPOCH = 0
EPOCH_START = 0
VALIDATION_SET_SIZE = 0.2 # Proportion of training data to use as validation set
NUM_MODELS_TO_TRAIN = 1

if not USE_FNC_DATA and not USE_SNLI_DATA and not USE_FEVER_DATA:
    raise Exception("Must choose data to use")

# Number of extra samples used is EXTRA_SAMPLES_PER_EPOCH * FNC_TRAIN_SIZE
EXTRA_SAMPLES_PER_EPOCH = 1

RATIO_LOSS = 0.5

import random

# Model parameters
RAND0 = random.Random(0)
RAND1 = random.Random(1)
RAND2 = random.Random(2)
MAX_FEATURES = 5000
TARGET_SIZE = 4
DOMAIN_TARGET_SIZE = 3
HIDDEN_SIZE = 100
DOMAIN_HIDDEN_SIZE = 100
LABEL_HIDDEN_SIZE = None
TRAIN_KEEP_PROB = 0.6
L2_ALPHA = 0.01
CLIP_RATIO = 5
BATCH_SIZE = 100
LR_FACTOR = 0.01

# CNN parameters
FILTER_SIZES = [2, 3, 4]
NUM_FILTERS = 128
CNN_HEADLINE_LENGTH = 50
CNN_BODY_LENGTH = 500

# LABEL MAPPINGS
FNC_LABELS = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
FNC_LABELS_REV = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
SNLI_LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}
DOMAIN_MAPPING = {'FNC': 0, 'SNLI': 1, 'FEVER': 2}

# CNN feature paramters
EMBEDDING_PATH = "data/GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM = 300

# File paths
FNC_TRAIN_STANCES = "data/fnc_data/train_stances.csv"
FNC_TRAIN_BODIES = "data/fnc_data/train_bodies.csv"
FNC_TEST_STANCES = "data/fnc_data/competition_test_stances.csv"
FNC_TEST_BODIES = "data/fnc_data/competition_test_bodies.csv"

SNLI_TRAIN = 'data/snli_data/snli_1.0_train.jsonl' 
SNLI_VAL = 'data/snli_data/snli_1.0_dev.jsonl'
SNLI_TEST = 'data/snli_data/snli_1.0_test.jsonl'

FEVER_TRAIN = "data/fever_data/train.jsonl"
FEVER_WIKI = "data/fever_data/wiki-pages"
FEVER_DEV = "data/fever_data/shared_task_dev.jsonl"

