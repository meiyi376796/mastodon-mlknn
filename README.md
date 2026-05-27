# mastodon-mlknn

Multi-label user classification on Mastodon based on the **ML-KNN** algorithm with heterogeneous network features and overlapping community detection.

This project reproduces and extends the method described in:

> Huang, A., Xu, R., Chen, Y., & Guo, M. (2023). *Research on multi-label user classification of social media based on ML-KNN algorithm.* Technological Forecasting & Social Change, 188, 122271.

## Overview

The pipeline assigns multiple topic labels to Mastodon users by combining content, relational, and network-structural features:

1. **Data Collection** — Fetches seed users, their posts, follows/mentions/boosts from public Mastodon instances.
2. **Network Construction** — Builds a heterogeneous user–entity graph and computes topic-specific personalized PageRank scores.
3. **Feature Extraction**
   - **Topic features**: LDA over concatenated status text
   - **Overlapping community features**: label-propagation-based community detection using node importance (NI) and LDA topic similarity
   - **Relational features**: 8-D user-relation and 6-D entity-relation statistics per topic, driven by RS scores
   - **Graph embeddings**: node2vec on the heterogeneous graph
   - **Profile features**: log-scaled follower/following/status counts
4. **Classification** — ML-KNN with confidence-weighted training, feature selection via mutual information, and nested cross-validation.
5. **Baseline Comparison** — Evaluates against Binary Relevance SVM, Label Powerset SVM, MROC, and EdgeCluster.

## Project Structure

```
mastodon-mlknn/
├── main.py                # Pipeline entry point (fetch → network → features → train)
├── config.py              # Hyperparameters, data paths, topic sets
├── data_fetcher.py        # Mastodon API data collection with checkpoint/resume
├── network_builder.py     # Heterogeneous network construction & PageRank scoring
├── feature_extractor.py   # LDA, community detection, node2vec, feature assembly
├── ml_knn.py              # ML-KNN implementation, training, evaluation, nested CV
├── compare_baselines.py   # Baseline models (BR, LP, MROC, EdgeCluster)
├── logger.py              # Colored terminal logger
├── requirements.txt       # Python dependencies
├── data/                  # Intermediate artifacts (.pkl, .npz)
├── results/               # JSON result files
└── LICENSE                # MIT License
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

Run individual stages:

```bash
python main.py fetch      # Collect data from Mastodon API
python main.py network    # Build heterogeneous networks
python main.py features   # Extract features
python main.py train      # Train ML-KNN and evaluate
```

Run baseline comparison:

```bash
python compare_baselines.py
```

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `K_NEIGHBORS` | 10 | Number of nearest neighbors for ML-KNN |
| `SMOOTHING_FACTOR` | 1.0 | Laplace smoothing factor |
| `LDA_TOPICS` | 10 | LDA topic components |
| `GRAPH_EMB_DIM` | 32 | node2vec embedding dimension |
| `PROFILE_DIM` | 4 | Profile feature dimension |
| `SEED_USERS_PER_TOPIC` | 60 | Seed users per topic |
| `POSTS_PER_SEED` | 200 | Posts fetched per seed user |
| `N_OUTER_CV` | 5 | Outer CV folds |
| `N_INNER_CV` | 5 | Inner CV folds |

## Datasets

Two datasets with different topic combinations:

- **Dataset 1**: politics, technology, economy
- **Dataset 2**: economy, education, sports

## Metrics

Reported per experiment and nested CV:

- Hamming Loss
- Accuracy (subset accuracy)
- Precision (sample-based)
- Recall (sample-based)
- F1 Score (sample-based)

## License

MIT — see [LICENSE](LICENSE).
