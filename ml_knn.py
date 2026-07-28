"""ML-KNN training and evaluation utilities for multi-label classification."""

import json
import os
import pickle
import time
import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import (
    K_NEIGHBORS, SMOOTHING_FACTOR, RANDOM_SEED,
    DATA_DIR, RESULTS_DIR, METRIC_ORDER,
    N_OUTER_CV, N_INNER_CV,
    USER_REL_DIM, ENTITY_REL_DIM,
    DATASETS,
)
from logger import Logger

from feature_extractor import recompute_rel_features

log = Logger("mlknn")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

os.makedirs(RESULTS_DIR, exist_ok=True)

METRIC_LABELS = {"hamming_loss": "hamming", "accuracy": "acc", "precision": "prec",
                 "recall": "rec", "f1_score": "f1"}


def _powerset_labels(y: np.ndarray) -> np.ndarray:
    """Encode multi-label rows as integer powerset labels for stratification."""
    return y.dot(1 << np.arange(y.shape[1]))


def _stratified_kfold_splits(n_splits, x, y, shuffle=True, random_state=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Multi-label stratified KFold splits with fallback to ordinary KFold."""
    powerset = _powerset_labels(y)
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(skf.split(x, powerset))
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(kf.split(x))


def _stratified_train_test_split(y: np.ndarray, test_size: float = 0.3):
    """Split row indices stratified by powerset labels, with a plain-split fallback."""
    idx_all = np.arange(y.shape[0])
    try:
        return train_test_split(
            idx_all, test_size=test_size, random_state=RANDOM_SEED,
            stratify=_powerset_labels(y),
        )
    except ValueError:
        return train_test_split(
            idx_all, test_size=test_size, random_state=RANDOM_SEED,
        )


class MLKNN:
    """ML-KNN implementation with optional confidence-weighted labels."""

    def __init__(self, k: int = K_NEIGHBORS, s: float = SMOOTHING_FACTOR):
        self.k = k
        self.s = s
        self._fitted = False
        self.prior_1: np.ndarray | None = None
        self.prior_0: np.ndarray | None = None
        self.posterior_1: np.ndarray | None = None
        self.posterior_0: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._effective_k: int | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            label_confidence: np.ndarray | None = None):
        """Estimate label priors and neighbor-count posteriors from training data."""
        n_samples, n_labels = y_train.shape
        effective_k = max(1, min(self.k, n_samples - 1))

        # Confidence weighting gives heuristic labels lower influence in training.
        if label_confidence is not None:
            sample_weight = np.clip(label_confidence.mean(axis=1), 0.3, 1.0)
        else:
            sample_weight = np.ones(n_samples)

        total_weight = sample_weight.sum()
        y_sums = (y_train * sample_weight[:, None]).sum(axis=0)
        self.prior_1 = (self.s + y_sums) / (self.s * 2 + total_weight)
        self.prior_0 = 1.0 - self.prior_1

        # For each label, count how many of a sample's k neighbors have that label.
        distances = cosine_distances(x_train)
        np.fill_diagonal(distances, np.inf)
        k_nearest = np.argpartition(distances, effective_k, axis=1)[:, :effective_k]

        posterior_1 = np.zeros((n_labels, effective_k + 1))
        posterior_0 = np.zeros((n_labels, effective_k + 1))

        for l in range(n_labels):
            c = np.zeros(effective_k + 1)
            c_prime = np.zeros(effective_k + 1)
            for i in range(n_samples):
                delta = int(y_train[k_nearest[i], l].sum())
                delta = min(delta, effective_k)
                if label_confidence is not None:
                    if y_train[i, l] == 1:
                        c[delta] += label_confidence[i, l]
                    else:
                        c_prime[delta] += 1.0 - label_confidence[i, l]
                elif y_train[i, l] == 1:
                    c[delta] += 1.0
                else:
                    c_prime[delta] += 1.0

            # Laplace smoothing prevents zero-probability decisions at prediction.
            posterior_1[l] = (self.s + c) / (self.s * (effective_k + 1) + c.sum())
            posterior_0[l] = (self.s + c_prime) / (self.s * (effective_k + 1) + c_prime.sum())

        self.posterior_1 = posterior_1
        self.posterior_0 = posterior_0

        self._X_train = x_train
        self._y_train = y_train
        self._effective_k = effective_k
        self._fitted = True

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict each label independently from neighbor-count likelihoods."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        n_test = x_test.shape[0]
        n_labels = len(self.prior_1)
        n_train = self._X_train.shape[0]
        effective_k = min(self._effective_k, n_train - 1)
        effective_k = max(1, effective_k)
        y_pred = np.zeros((n_test, n_labels), dtype=np.int32)

        distances = cosine_distances(x_test, self._X_train)
        k_nearest = np.argpartition(distances, effective_k, axis=1)[:, :effective_k]

        for t in range(n_test):
            for l in range(n_labels):
                c_t_l = int(self._y_train[k_nearest[t], l].sum())
                c_t_l = min(c_t_l, effective_k)
                prob_1 = self.prior_1[l] * self.posterior_1[l, c_t_l]
                prob_0 = self.prior_0[l] * self.posterior_0[l, c_t_l]
                y_pred[t, l] = 1 if prob_1 >= prob_0 else 0

        return y_pred


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, average: str = "sample") -> dict:
    """Compute multi-label metrics in the order expected by the reports."""
    if average not in {"sample", "micro"}:
        raise ValueError("average must be 'sample' or 'micro'")

    n = y_true.shape[0]

    hamming_loss = (y_true != y_pred).sum() / (n * y_true.shape[1])

    intersection = (y_pred * y_true).sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    true_sum = y_true.sum(axis=1)
    union = pred_sum + true_sum - intersection

    if average == "micro":
        total_intersection = intersection.sum()
        total_pred = pred_sum.sum()
        total_true = true_sum.sum()
        total_union = union.sum()
        precision = total_intersection / max(1, total_pred)
        recall = total_intersection / max(1, total_true)
        accuracy = total_intersection / max(1, total_union)
        f1 = 2 * total_intersection / max(1, total_true + total_pred)
    else:
        precision = np.mean([intersection[i] / max(1, pred_sum[i]) for i in range(n)])
        recall = np.mean([intersection[i] / max(1, true_sum[i]) for i in range(n)])
        accuracy = np.mean([intersection[i] / max(1, union[i]) for i in range(n)])
        f1 = np.mean([2 * intersection[i] / max(1, true_sum[i] + pred_sum[i]) for i in range(n)])

    metrics = {
        "hamming_loss": float(hamming_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    return {name: metrics[name] for name in METRIC_ORDER}


def select_features_mi(x_train: np.ndarray, y_train: np.ndarray, n_select: int = 80):
    """Select features with the largest summed mutual information over labels."""
    if x_train.shape[1] <= n_select:
        return np.arange(x_train.shape[1])
    mi_scores = np.zeros(x_train.shape[1])
    for l in range(y_train.shape[1]):
        if y_train[:, l].sum() > 1:
            mi_scores += mutual_info_classif(
                x_train, y_train[:, l], random_state=RANDOM_SEED
            )
    return np.argsort(mi_scores)[-n_select:]


def tune_hyperparameters(x: np.ndarray, y: np.ndarray, n_folds: int = 5,
                         k_values=None, s_values=None):
    """Grid-search ML-KNN hyperparameters with cross-validated F1."""
    if k_values is None:
        k_values = [3, 5, 8, 10, 12, 15, 20]
    if s_values is None:
        s_values = [0.5, 1.0, 2.0]

    best_score = -1
    best_params = {"k": K_NEIGHBORS, "s": SMOOTHING_FACTOR}

    n_splits = min(n_folds, len(x))
    splits = list(_stratified_kfold_splits(n_splits, x, y, shuffle=True, random_state=RANDOM_SEED))

    for k in k_values:
        if k < 1 or k >= len(x) - 1:
            continue
        for s in s_values:
            scores = []
            for train_idx, val_idx in splits:
                x_tr, x_val = x[train_idx], x[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = MLKNN(k=k, s=s)
                model.fit(x_tr, y_tr)
                y_pred = model.predict(x_val)
                metrics = evaluate(y_val, y_pred)
                scores.append(metrics["f1_score"])

            avg_f1 = np.mean(scores)
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = {"k": k, "s": s}

    return best_params


def _load_labeled_data(dataset_name: str):
    """Load feature matrix and return the labeled subset."""
    features_path = os.path.join(DATA_DIR, f"{dataset_name}_features.npz")
    if not os.path.exists(features_path):
        log.skip(f"{features_path} not found")
        return None

    data_npz = np.load(features_path, allow_pickle=True)
    x = data_npz["X"]
    y = data_npz["y"]
    topics = [str(t) for t in data_npz["topics"]]
    label_scores = data_npz.get("label_scores", None)
    log.info(f"data · x={x.shape} y={y.shape} topics={len(topics)}")

    labeled_mask = y.sum(axis=1) > 0
    x_labeled = x[labeled_mask]
    y_labeled = y[labeled_mask]
    if label_scores is not None:
        label_scores = label_scores[labeled_mask]

    if x_labeled.shape[0] < 20:
        log.error("fewer than 20 labeled users")
        return None

    user_ids_all = data_npz["user_ids"]
    labeled_uids = [str(uid) for uid in user_ids_all[labeled_mask]]

    return x_labeled, y_labeled, topics, label_scores, x.shape[0], labeled_uids


def _per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, topics: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-label accuracy, precision, recall, f1."""
    result = {}
    for l, topic in enumerate(topics):
        tp = ((y_pred[:, l] == 1) & (y_true[:, l] == 1)).sum()
        fp = ((y_pred[:, l] == 1) & (y_true[:, l] == 0)).sum()
        fn = ((y_pred[:, l] == 0) & (y_true[:, l] == 1)).sum()
        l_acc = (y_true[:, l] == y_pred[:, l]).mean()
        l_prec = tp / max(1, tp + fp)
        l_rec = tp / max(1, tp + fn)
        l_f1 = 2 * l_prec * l_rec / max(1e-9, l_prec + l_rec)
        result[topic] = {
            "accuracy": float(l_acc),
            "precision": float(l_prec),
            "recall": float(l_rec),
            "f1_score": float(l_f1),
        }
    return result


def _recompute_pagerank_features(
    network,
    labeled_uids: list[str],
    train_idx: np.ndarray,
    topics: list[str],
    x_labeled: np.ndarray,
) -> np.ndarray:
    """Recompute PageRank-based relational features using current train fold."""
    if network is None:
        return x_labeled

    train_uids = {labeled_uids[int(i)] for i in train_idx}
    network.set_train_fold(train_uids)
    network.compute_rs_scores()
    new_user_rel, new_entity_rel = recompute_rel_features(network, labeled_uids, topics)
    n_topics = len(topics)
    ur_end = USER_REL_DIM * n_topics
    er_end = ur_end + ENTITY_REL_DIM * n_topics
    x_labeled_fixed = x_labeled.copy()
    x_labeled_fixed[:, :ur_end] = new_user_rel
    x_labeled_fixed[:, ur_end:er_end] = new_entity_rel
    return x_labeled_fixed


def _scale_and_select(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    n_select: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale features and select the top features by mutual information."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    selected = select_features_mi(x_train_scaled, y_train, n_select=n_select)
    if len(selected) < x_train_scaled.shape[1]:
        x_train_scaled = x_train_scaled[:, selected]
        x_test_scaled = x_test_scaled[:, selected]
    return x_train_scaled, x_test_scaled, selected


def _save_results(results: dict, results_path: str) -> None:
    """Serialize results to JSON and log the file path."""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"file · {results_path}")


def run_experiment(dataset_name: str):
    """Run train-test evaluation for a saved feature matrix."""
    log.info(f"▸ {dataset_name}")
    start_time = time.time()

    loaded = _load_labeled_data(dataset_name)
    if loaded is None:
        return None
    x_labeled, y_labeled, topics, label_scores, n_users, labeled_uids = loaded

    idx_train, idx_test = _stratified_train_test_split(y_labeled)

    x_train, x_test = x_labeled[idx_train], x_labeled[idx_test]
    y_train, y_test = y_labeled[idx_train], y_labeled[idx_test]
    log.info(f"split · train={x_train.shape[0]} test={x_test.shape[0]}")

    # Recompute PageRank-based features using only training-fold seeds.
    net_path = os.path.join(DATA_DIR, f"{dataset_name}_network.pkl")
    network = None
    if os.path.exists(net_path):
        with open(net_path, "rb") as f:
            network = pickle.load(f)

    x_labeled = _recompute_pagerank_features(
        network, labeled_uids, idx_train, topics, x_labeled
    )
    x_train = x_labeled[idx_train]
    x_test = x_labeled[idx_test]

    x_train, x_test, selected = _scale_and_select(x_train, x_test, y_train)
    log.info(f"features · selected={len(selected)}/{x_labeled.shape[1]}")

    log.info("tuning …")
    params = tune_hyperparameters(x_train, y_train)
    log.info(f"params · k={params['k']} s={params['s']}")

    lc = None
    if label_scores is not None:
        lc = np.clip(label_scores[idx_train], 0.0, 1.0)

    fit_start = time.time()
    model = MLKNN(k=params['k'], s=params['s'])
    model.fit(x_train, y_train, label_confidence=lc)
    train_time = time.time() - fit_start
    log.info(f"training · {train_time:.1f}s")

    y_pred = model.predict(x_test)

    metrics = evaluate(y_test, y_pred)
    for name in METRIC_ORDER:
        label = METRIC_LABELS[name]
        log.info(f"{label} · {metrics[name]:.4f}")

    per_label = _per_label_metrics(y_test, y_pred, topics)

    results = {
        "dataset": dataset_name,
        "topics": [str(t) for t in topics],
        "n_users": n_users,
        "n_labeled": int(x_labeled.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_labeled.shape[1]),
        "best_k": params["k"],
        "best_s": params["s"],
        "train_time_s": train_time,
        "metrics": metrics,
        "per_label": per_label,
    }

    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    _save_results(results, results_path)
    log.info(f"total · {time.time() - start_time:.1f}s")

    return metrics


def run_nested_cv(dataset_name: str, n_outer: int = N_OUTER_CV,
                  n_inner: int = N_INNER_CV):
    """Run nested cross-validation and report mean ± std across outer folds."""
    log.info(f"▸ {dataset_name} (nested CV)")
    start_time = time.time()

    loaded = _load_labeled_data(dataset_name)
    if loaded is None:
        return None
    x_labeled, y_labeled, topics, label_scores, n_users, labeled_uids = loaded

    n_samples = x_labeled.shape[0]
    n_outer = min(n_outer, n_samples)

    # Load network once; rs_scores are recomputed per fold below.
    net_path = os.path.join(DATA_DIR, f"{dataset_name}_network.pkl")
    network = None
    if os.path.exists(net_path):
        with open(net_path, "rb") as f:
            network = pickle.load(f)

    fold_metrics = []
    fold_per_label = []
    fold_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        _stratified_kfold_splits(n_outer, x_labeled, y_labeled, shuffle=True, random_state=RANDOM_SEED)
    ):
        x_labeled_fold = _recompute_pagerank_features(
            network, labeled_uids, train_idx, topics, x_labeled
        )
        x_tr, x_te = x_labeled_fold[train_idx], x_labeled_fold[test_idx]
        y_tr, y_te = y_labeled[train_idx], y_labeled[test_idx]

        x_tr, x_te, _ = _scale_and_select(x_tr, x_te, y_tr)

        params = tune_hyperparameters(x_tr, y_tr, n_folds=n_inner)

        lc = None
        if label_scores is not None:
            lc = np.clip(label_scores[train_idx], 0.0, 1.0)

        model = MLKNN(k=params["k"], s=params["s"])
        model.fit(x_tr, y_tr, label_confidence=lc)
        y_pred = model.predict(x_te)

        metrics = evaluate(y_te, y_pred)
        fold_metrics.append(metrics)
        fold_params.append(params)
        fold_per_label.append(_per_label_metrics(y_te, y_pred, topics))

        log.info(f"fold {fold_idx + 1}/{n_outer} · "
                 f"train={x_tr.shape[0]} test={x_te.shape[0]} "
                 f"k={params['k']} s={params['s']} "
                 f"f1={metrics['f1_score']:.4f}")

    agg_metrics = {}
    for name in METRIC_ORDER:
        values = [m[name] for m in fold_metrics]
        agg_metrics[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }

    agg_per_label = {}
    for topic in topics:
        agg_per_label[topic] = {}
        for stat in ["accuracy", "precision", "recall", "f1_score"]:
            values = [pl[topic][stat] for pl in fold_per_label]
            agg_per_label[topic][stat] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }

    parts = []
    for name in METRIC_ORDER:
        entry = agg_metrics[name]
        parts.append(f"{METRIC_LABELS[name]}={entry['mean']:.4f}±{entry['std']:.4f}")
    log.info(f"nested CV · {' '.join(parts)}")

    results = {
        "dataset": dataset_name,
        "topics": [str(t) for t in topics],
        "n_users": n_users,
        "n_labeled": int(x_labeled.shape[0]),
        "n_outer_folds": n_outer,
        "n_inner_folds": n_inner,
        "n_features": int(x_labeled.shape[1]),
        "metrics": agg_metrics,
        "per_label": agg_per_label,
        "fold_params": fold_params,
        "total_time_s": time.time() - start_time,
    }

    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_nested_cv_results.json")
    _save_results(results, results_path)
    log.info(f"total · {time.time() - start_time:.1f}s")

    return agg_metrics


if __name__ == "__main__":
    total_start = time.time()
    log.header()
    log.info("args · all")
    for ds, _ in DATASETS:
        run_nested_cv(ds)
        run_experiment(ds)
    log.ok(f"train complete · {time.time() - total_start:.1f}s")
    log.blank()
