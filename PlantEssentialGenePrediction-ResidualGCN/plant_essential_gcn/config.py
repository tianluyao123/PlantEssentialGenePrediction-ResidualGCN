import os
from dataclasses import dataclass, asdict

import torch


@dataclass
class DeploymentConfig:

    K: int = 3
    D: int = 4
    SEQ_MAX_LEN: int = 1200
    PADDING_STRATEGY: str = "smart"

    REDUCED_MAIN_DIM: int = 512
    USE_FEATURE_REDUCTION: bool = True
    USE_BIO_FEATURES: bool = True
    CDS_FEATURE_DIM: int = 256
    AA_FEATURE_DIM: int = 128
    BIO_FEATURE_DIM: int = 158

    GCN_HIDDEN: int = 512
    GCN_LAYERS: int = 3
    DROPOUT: float = 0.3

    pos_weight: float = 4.23
    optimal_threshold: float = 0.5

    model_filename: str = "model.pth"
    reducer_main_filename: str = "reducer_main.pkl"
    reducer_bio_filename: str = "reducer_bio.pkl"
    metadata_filename: str = "metadata.json"
    config_filename: str = "config.json"

    def to_dict(self):
        return asdict(self)


class OptimizedConfig:

    N_FOLDS = 5
    N_REPEATS = 5
    RANDOM_STATE = 42

    DATA_DIR = os.environ.get("PLANT_GCN_DATA_DIR", os.path.join("data", "arabidopsis"))

    ESSENTIAL_TRAIN_FASTA = os.path.join(DATA_DIR, "cds_essential_train.fasta")
    ESSENTIAL_TEST_FASTA = os.path.join(DATA_DIR, "cds_essential_test.fasta")
    NON_ESSENTIAL_TRAIN_FASTA = os.path.join(DATA_DIR, "cds_nonessential_train.fasta")
    NON_ESSENTIAL_TEST_FASTA = os.path.join(DATA_DIR, "cds_nonessential_test.fasta")

    ESSENTIAL_AA_TRAIN_FASTA = os.path.join(DATA_DIR, "aa_essential_train.fasta")
    ESSENTIAL_AA_TEST_FASTA = os.path.join(DATA_DIR, "aa_essential_test.fasta")
    NON_ESSENTIAL_AA_TRAIN_FASTA = os.path.join(DATA_DIR, "aa_nonessential_train.fasta")
    NON_ESSENTIAL_AA_TEST_FASTA = os.path.join(DATA_DIR, "aa_nonessential_test.fasta")

    SAVE_BASE_PATH = os.environ.get("PLANT_GCN_OUTPUT_DIR", "outputs")
    DEPLOYMENT_PATH = os.path.join(SAVE_BASE_PATH, "deployment_models")

    K = 3
    D = 4
    SEQ_MAX_LEN = 1200
    PADDING_STRATEGY = "smart"

    REDUCED_MAIN_DIM = 512
    USE_FEATURE_REDUCTION = True

    USE_BIO_FEATURES = True
    USE_CDS_FEATURES = True
    USE_AA_FEATURES = True
    CDS_FEATURE_DIM = 256
    AA_FEATURE_DIM = 128
    BIO_FEATURE_DIM = 158

    DROPOUT = 0.3
    LSTM_LAYERS = 2
    WEIGHT_DECAY = 1e-5
    GCN_HIDDEN = 512
    GCN_LAYERS = 3
    LSTM_HIDDEN = 128
    ATTENTION_HEADS = 8
    MIN_KMER_COUNT = 2

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    LR = 8e-5
    WARMUP_EPOCHS = 8
    WARMUP_START_FACTOR = 0.2
    MAX_EPOCHS = 200
    EARLY_STOP_PATIENCE = 30

    POS_WEIGHT_CAP = 3.0
    POS_WEIGHT_MULTIPLIER = 1.0

    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.20
    FOCAL_GAMMA = 2.0

    AUGMENT_TRAIN = True
    AUGMENT_PROB = 0.5

    LABEL_SMOOTHING = 0.01
    FN_PENALTY = 1.0

    OPTIMIZATION_METRIC = "youden_j"
    SP_FLOOR = 0.50

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
