"""Shared configuration for data collection, feature extraction, and reporting."""

# Mastodon instances used for hashtag timeline sampling.
INSTANCES = [
    "https://mastodon.social",
    "https://mastodon.online",
    "https://fosstodon.org",
    "https://mastodon.world",
    "https://mstdn.social",
    "https://mas.to",
]

DATASET1_TOPICS = ["politics", "technology", "economy"]
DATASET2_TOPICS = ["economy", "education", "sports"]

# (dataset name, topics) pairs shared by every pipeline stage.
DATASETS = [("dataset1", DATASET1_TOPICS), ("dataset2", DATASET2_TOPICS)]

# Data collection limits. Higher values improve coverage but increase API time.
SEED_USERS_PER_TOPIC = 60
POSTS_PER_SEED = 200

# ML-KNN defaults; training can tune around these values.
K_NEIGHBORS = 10
SMOOTHING_FACTOR = 1.0

# Feature and community-detection dimensions.
LDA_TOPICS = 10
MIN_COMMUNITY_SIZE = 3
USER_REL_DIM = 8
ENTITY_REL_DIM = 6
GRAPH_EMB_DIM = 32
PROFILE_DIM = 4

# Metric order shared by console output and JSON summaries.
METRIC_ORDER = ["hamming_loss", "accuracy", "precision", "recall", "f1_score"]
METRIC_LABELS = {
    "hamming_loss": "Hamming Loss",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1",
}

RANDOM_SEED = 42

N_OUTER_CV = 5
N_INNER_CV = 5

DATA_DIR = "data"
RESULTS_DIR = "results"
