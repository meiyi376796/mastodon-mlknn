"""Compare the proposed feature set against SVM and network-community baselines."""

import os
import json
import time
import pickle

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED,
    USER_REL_DIM, ENTITY_REL_DIM, LDA_TOPICS, GRAPH_EMB_DIM, PROFILE_DIM,
)
from ml_knn import (
    MLKNN, evaluate, select_features_mi, tune_hyperparameters,
    _powerset_labels, _stratified_kfold_splits,
)
from feature_extractor import recompute_rel_features
from logger import Logger

log = Logger("compare")

os.makedirs(RESULTS_DIR, exist_ok=True)

def comparison_metrics(y_true, y_pred):
    """Compute multi-label metrics with sample-based averaging."""
    return evaluate(y_true, y_pred)


class BinaryRelevanceSVM:
    """Train one binary SVM per label."""

    def __init__(self, C=1.0, kernel="rbf", class_weight=None):
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        self.clfs = []

    def fit(self, X, y):
        self.clfs = []
        for l in range(y.shape[1]):
            classes = np.unique(y[:, l])
            if len(classes) < 2:
                # Some training splits contain only one class for a label.
                self.clfs.append(int(classes[0]))
                continue
            clf = SVC(kernel=self.kernel, C=self.C, class_weight=self.class_weight, random_state=RANDOM_SEED)
            self.clfs.append(clf.fit(X, y[:, l]))

    def predict(self, X):
        preds = []
        for clf in self.clfs:
            if isinstance(clf, int):
                preds.append(np.full(X.shape[0], clf, dtype=np.int32))
            else:
                preds.append(clf.predict(X).astype(np.int32))
        return np.column_stack(preds).astype(np.int32)


class LabelPowersetSVM:
    """Treat each unique label combination as one multiclass target."""

    def __init__(self, C=1.0):
        self.C = C
        self.clf = None
        self._combo_to_class = {}
        self._class_to_combo = {}

    def fit(self, X, y):
        y_int = y.astype(np.int32)
        combos = sorted(set(tuple(row) for row in y_int))
        self._combo_to_class = {c: i for i, c in enumerate(combos)}
        self._class_to_combo = {i: np.array(c, dtype=np.int32) for i, c in enumerate(combos)}
        y_class = np.array([self._combo_to_class[tuple(row)] for row in y_int])
        self.clf = SVC(kernel="rbf", C=self.C, random_state=RANDOM_SEED)
        self.clf.fit(X, y_class)

    def predict(self, X):
        y_class = self.clf.predict(X).astype(np.int32)
        n_labels = len(next(iter(self._class_to_combo.values())))
        y_pred = np.zeros((X.shape[0], n_labels), dtype=np.int32)
        for i, c in enumerate(y_class):
            if c in self._class_to_combo:
                y_pred[i] = self._class_to_combo[c]
        return y_pred


def build_user_topology_graph(network, user_ids):
    """Project the heterogeneous graph down to user-user topology only."""
    user_set = set(user_ids)
    topo = nx.Graph()
    topo.add_nodes_from(user_ids)

    for edge in network.graph.edges():
        u, v = edge[0], edge[1]
        if network.graph.nodes[u].get("type") != "user":
            continue
        if network.graph.nodes[v].get("type") != "user":
            continue
        uid_u = network.idx_to_user.get(u)
        uid_v = network.idx_to_user.get(v)
        if uid_u in user_set and uid_v in user_set and uid_u != uid_v:
            topo.add_edge(uid_u, uid_v, weight=topo.get_edge_data(uid_u, uid_v, {}).get("weight", 0) + 1)

    return topo


def _add_community(communities, seen, members, min_size=2):
    """Add a community once, ignoring small or duplicate member sets."""
    members = frozenset(members)
    if len(members) < min_size or members in seen:
        return
    seen.add(members)
    communities.append(members)


def build_mroc_features(network, user_ids):
    """Build overlapping community membership features for the MROC baseline."""
    topo = build_user_topology_graph(network, user_ids)
    n_users = len(user_ids)

    if topo.number_of_edges() == 0:
        return np.zeros((n_users, 1), dtype=np.float32), {
            "n_communities": 1,
            "n_topology_edges": 0,
        }

    communities = []
    seen = set()

    # Multiple resolutions approximate overlapping memberships from modularity runs.
    for resolution in (0.5, 0.8, 1.0, 1.5, 2.0):
        detected = nx.algorithms.community.greedy_modularity_communities(
            topo,
            weight="weight",
            resolution=resolution,
        )
        for comm in detected:
            _add_community(communities, seen, comm)

    for component in nx.connected_components(topo):
        _add_community(communities, seen, component)

    if not communities:
        return np.zeros((n_users, 1), dtype=np.float32), {
            "n_communities": 1,
            "n_topology_edges": topo.number_of_edges(),
        }

    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    features = np.zeros((n_users, len(communities)), dtype=np.float32)
    for j, comm in enumerate(communities):
        for uid in comm:
            row = uid_to_row.get(uid)
            if row is not None:
                features[row, j] = 1.0

    return features, {
        "n_communities": len(communities),
        "n_topology_edges": topo.number_of_edges(),
    }


def choose_edge_cluster_count(n_edges):
    """Scale edge clusters with graph size while keeping KMeans bounded."""
    if n_edges < 2:
        return 1
    return min(64, max(2, int(np.sqrt(n_edges) * 2)))


def build_edgecluster_features(network, user_ids, n_clusters=None):
    """Represent users by normalized memberships in clustered graph edges."""
    topo = build_user_topology_graph(network, user_ids)
    n_users = len(user_ids)
    edges = list(topo.edges())

    if len(edges) < 2:
        return np.zeros((n_users, 1), dtype=np.float32), {
            "n_edge_clusters": 1,
            "n_topology_edges": len(edges),
        }

    if n_clusters is None:
        n_clusters = choose_edge_cluster_count(len(edges))
    n_clusters = min(max(2, n_clusters), len(edges))

    uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    edge_matrix = np.zeros((len(edges), n_users), dtype=np.float32)
    for i, (u, v) in enumerate(edges):
        edge_matrix[i, uid_to_row[u]] = 1.0
        edge_matrix[i, uid_to_row[v]] = 1.0

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    edge_labels = km.fit_predict(edge_matrix)

    features = np.zeros((n_users, n_clusters), dtype=np.float32)
    for (u, v), label in zip(edges, edge_labels):
        features[uid_to_row[u], label] += 1.0
        features[uid_to_row[v], label] += 1.0

    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    features = features / row_sum

    return features, {
        "n_edge_clusters": n_clusters,
        "n_topology_edges": len(edges),
    }


def tune_svm(cls, X, y, n_folds=3, c_values=None, **kw):
    """Cross-validate the SVM regularization parameter for a baseline class."""
    if c_values is None:
        c_values = [0.1, 1.0, 10.0]

    best_c = c_values[0]
    best_f1 = -1.0
    n_splits = min(n_folds, len(X))
    splits = list(_stratified_kfold_splits(n_splits, X, y, shuffle=True, random_state=RANDOM_SEED))

    for c in c_values:
        scores = []
        for tr_idx, val_idx in splits:
            m = cls(C=c, **kw)
            m.fit(X[tr_idx], y[tr_idx])
            yp = m.predict(X[val_idx])
            scores.append(evaluate(y[val_idx], yp)["f1_score"])
        avg_f1 = np.mean(scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_c = c

    return best_c


def run(dataset_name):
    """Run all baseline models and write a comparison JSON report."""
    log.info(f"▸ {dataset_name}")
    t0 = time.time()

    fpath = os.path.join(DATA_DIR, f"{dataset_name}_features.npz")
    npath = os.path.join(DATA_DIR, f"{dataset_name}_network.pkl")
    if not os.path.exists(fpath):
        log.skip(f"{fpath} not found")
        return

    data_npz = np.load(fpath, allow_pickle=True)
    Xf, yf = data_npz["X"], data_npz["y"]
    label_scores = data_npz.get("label_scores", None)
    user_ids = [str(uid) for uid in data_npz["user_ids"]]
    mask = yf.sum(1) > 0
    X, y = Xf[mask], yf[mask]
    ls = label_scores[mask] if label_scores is not None else None
    log.info(f"data · X={X.shape} samples={len(y)} positives={y.sum()} labels={y.shape[1]}")

    if len(y) < 20:
        log.warn(f"{dataset_name} · too few labeled samples ({len(y)}), will skip")
        return

    if not os.path.exists(npath):
        log.skip(f"{npath} not found")
        return
    with open(npath, "rb") as f:
        network = pickle.load(f)

    log.info(f"network · nodes={network.graph.number_of_nodes()} edges={network.graph.number_of_edges()}")

    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=_powerset_labels(y),
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=RANDOM_SEED,
        )
    ytr, yte = y[train_idx], y[test_idx]

    # Recompute PageRank-based features using only training-fold seeds.
    labeled_uids = [user_ids[i] for i in range(len(user_ids)) if mask[i]]
    train_uids = {labeled_uids[i] for i in train_idx}
    network.set_train_fold(train_uids)
    network.compute_rs_scores()
    n_topics = y.shape[1]
    topics_list = [str(t) for t in data_npz["topics"]]
    new_user_rel, new_entity_rel = recompute_rel_features(network, labeled_uids, topics_list)
    ur_end = USER_REL_DIM * n_topics
    er_end = ur_end + ENTITY_REL_DIM * n_topics
    X_fixed = X.copy()
    X_fixed[:, :ur_end] = new_user_rel
    X_fixed[:, ur_end:er_end] = new_entity_rel
    X = X_fixed

    def split_scale(features):
        scaler = StandardScaler()
        return scaler.fit_transform(features[train_idx]), scaler.transform(features[test_idx])

    Xtr, Xte = split_scale(X)
    selected = select_features_mi(Xtr, ytr, n_select=80)
    if len(selected) < Xtr.shape[1]:
        Xtr, Xte = Xtr[:, selected], Xte[:, selected]

    n_topics = y.shape[1]
    lda_dim = int(data_npz.get("lda_dim", LDA_TOPICS))
    simple_leading = (USER_REL_DIM + ENTITY_REL_DIM) * n_topics
    simple_trailing = GRAPH_EMB_DIM + PROFILE_DIM + lda_dim
    # The "simple" baselines exclude community and graph-embedding features.
    X_simple = np.hstack([X[:, :simple_leading], X[:, -simple_trailing:-GRAPH_EMB_DIM]])
    Xstr, Xste = split_scale(X_simple)

    Xc_full, mroc_meta = build_mroc_features(network, user_ids)
    Xc = Xc_full[mask]
    Xctr, Xcte = split_scale(Xc)
    log.info(f"MROC · communities={mroc_meta['n_communities']} edges={mroc_meta['n_topology_edges']}")

    Xe_full, edge_meta = build_edgecluster_features(network, user_ids)
    Xe = Xe_full[mask]
    Xetr, Xete = split_scale(Xe)
    log.info(f"EdgeCluster · clusters={edge_meta['n_edge_clusters']} edges={edge_meta['n_topology_edges']}")

    results = {}

    log.info("MLUCHNCD · training …")
    params = tune_hyperparameters(Xtr, ytr, n_folds=3, k_values=[3, 5, 8, 10], s_values=[0.5, 1.0])
    k, s = params["k"], params["s"]
    t_start = time.time()
    m = MLKNN(k=k, s=s)
    lc = np.clip(ls[train_idx], 0.0, 1.0) if ls is not None else None
    m.fit(Xtr, ytr, label_confidence=lc)
    yp_ml = m.predict(Xte)
    results["MLUCHNCD"] = {
        **comparison_metrics(yte, yp_ml),
        "train_s": time.time() - t_start,
        "k": k,
        "s": s,
    }

    log.info("BR · training …")
    c_br = tune_svm(BinaryRelevanceSVM, Xstr, ytr)
    t_start = time.time()
    m = BinaryRelevanceSVM(C=c_br)
    m.fit(Xstr, ytr)
    results["BR"] = {
        **comparison_metrics(yte, m.predict(Xste)),
        "train_s": time.time() - t_start,
        "C": c_br,
    }

    log.info("LP · training …")
    c_lp = tune_svm(LabelPowersetSVM, Xstr, ytr)
    t_start = time.time()
    m = LabelPowersetSVM(C=c_lp)
    m.fit(Xstr, ytr)
    results["LP"] = {
        **comparison_metrics(yte, m.predict(Xste)),
        "train_s": time.time() - t_start,
        "C": c_lp,
    }

    log.info("MROC · training …")
    c_mroc = tune_svm(BinaryRelevanceSVM, Xctr, ytr, c_values=[0.01, 0.1, 1.0, 10.0], kernel="linear", class_weight="balanced")
    t_start = time.time()
    m = BinaryRelevanceSVM(C=c_mroc, kernel="linear", class_weight="balanced")
    m.fit(Xctr, ytr)
    results["MROC"] = {
        **comparison_metrics(yte, m.predict(Xcte)),
        "train_s": time.time() - t_start,
        "C": c_mroc,
        **mroc_meta,
    }

    log.info("EdgeCluster · training …")
    c_ec = tune_svm(BinaryRelevanceSVM, Xetr, ytr, c_values=[0.01, 0.1, 1.0, 10.0], kernel="linear", class_weight="balanced")
    t_start = time.time()
    m = BinaryRelevanceSVM(C=c_ec, kernel="linear", class_weight="balanced")
    m.fit(Xetr, ytr)
    results["EdgeCluster"] = {
        **comparison_metrics(yte, m.predict(Xete)),
        "train_s": time.time() - t_start,
        "C": c_ec,
        **edge_meta,
    }

    for a, r in results.items():
        log.info(f"{a} · f1={r['f1_score']:.4f}")

    sp = os.path.join(RESULTS_DIR, f"{dataset_name}_comparison.json")
    with open(sp, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"file · {sp}")
    log.info(f"total · {time.time() - t0:.1f}s")


if __name__ == "__main__":
    t0 = time.time()
    log.header()
    log.info("args · all")
    run("dataset1")
    run("dataset2")
    log.ok(f"compare complete · {time.time() - t0:.1f}s")
    log.blank()
