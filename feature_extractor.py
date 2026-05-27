"""Feature extraction for Mastodon multi-label user classification."""

import os
import pickle
import math
import time
import re
from collections import defaultdict
from contextlib import redirect_stderr

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import (
    DATA_DIR, DATASET1_TOPICS, DATASET2_TOPICS,
    LDA_TOPICS, MIN_COMMUNITY_SIZE, RANDOM_SEED,
    GRAPH_EMB_DIM, PROFILE_DIM,
)
from logger import Logger

log = Logger("features")


def _clean_text(text: str) -> str:
    """Normalize Mastodon HTML status content for vectorization."""
    if not text:
        return ""

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^\w\s#@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_user_texts(seed_posts: dict) -> dict[str, str]:
    """Concatenate each user's statuses into one document for topic modeling."""
    user_texts = {}
    for uid, posts in seed_posts.items():
        if not posts:
            continue
        texts = []
        for p in posts[:300]:
            content = p.get("content", "")
            if isinstance(content, str):
                texts.append(_clean_text(content))
        combined = " ".join(texts)
        if len(combined) > 50:
            user_texts[uid] = combined
    return user_texts


def run_lda(user_texts: dict[str, str]) -> dict[str, np.ndarray]:
    """Infer a fixed-width topic distribution for each user document."""
    uids = list(user_texts.keys())
    texts = [user_texts[uid] for uid in uids]

    if len(texts) < 10:
        log.warn(f"too few documents for LDA ({len(texts)})")

        # Dirichlet draws preserve downstream shape on small datasets.
        rng = np.random.RandomState(RANDOM_SEED)
        user_topics = {}
        for uid in uids:
            user_topics[uid] = rng.dirichlet(np.ones(LDA_TOPICS))
        return user_topics

    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english",
        min_df=2, max_df=0.9
    )
    X = vectorizer.fit_transform(texts)

    if X.shape[1] == 0:
        log.warn(f"LDA vocabulary empty ({len(texts)})")
        rng = np.random.RandomState(RANDOM_SEED)
        user_topics = {}
        for uid in uids:
            user_topics[uid] = rng.dirichlet(np.ones(LDA_TOPICS))
        return user_topics

    # Use the smaller of configured topics, corpus size, and vocabulary size.
    n_topics = min(LDA_TOPICS, len(texts) // 2, X.shape[1])
    n_topics = max(2, n_topics)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        random_state=RANDOM_SEED,
        max_iter=20,
    )
    doc_topics = lda.fit_transform(X)

    user_topics = {}
    for i, uid in enumerate(uids):
        dist = doc_topics[i]
        if dist.sum() > 0:
            dist = dist / dist.sum()
        else:
            dist = np.ones(n_topics) / n_topics

        # Keep feature dimensionality stable when LDA produces fewer components.
        if len(dist) < LDA_TOPICS:
            padded = np.zeros(LDA_TOPICS)
            padded[:len(dist)] = dist
            dist = padded
        user_topics[uid] = dist

    log.info(f"LDA · {len(texts)} docs {n_topics} topics · {time.time() - t0:.1f}s")
    return user_topics


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with explicit zero-vector handling."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_ni_coefficient(graph, user_to_idx: dict) -> dict[str, float]:
    """Compute node importance from degree plus local triangles, scaled to 0.5-1.0."""
    ni = {}
    all_values = []
    user_vals = {}

    for uid, idx in user_to_idx.items():
        neighbors = list(set(graph.successors(idx)) | set(graph.predecessors(idx)))
        du = len(neighbors)

        tri = 0
        n_set = set(neighbors)
        # Triangle count adds a local cohesion signal beyond raw degree.
        for v in neighbors:
            for w in set(graph.successors(v)) | set(graph.predecessors(v)):
                if w in n_set and w != v:
                    tri += 1
        tri //= 2

        val = du + tri
        user_vals[uid] = val
        all_values.append(val)

    if not all_values or max(all_values) == min(all_values):
        return {uid: 0.5 for uid in user_vals}

    val_min = min(all_values)
    val_max = max(all_values)
    denom = val_max - val_min

    for uid, val in user_vals.items():
        ni[uid] = 0.5 + 0.5 * (val - val_min) / denom

    return ni


def overlapping_community_detection(
    graph, user_to_idx: dict, idx_to_user: dict,
    user_topics: dict, ni_coeffs: dict,
    max_iterations: int = 20,
) -> dict[str, set[int]]:
    """Detect overlapping communities with topic-similarity and NI-weighted propagation."""
    user_ids = list(user_to_idx.keys())
    n_users = len(user_ids)

    if n_users < 10:
        log.warn(f"too few users for communities ({n_users})")
        return {uid: set() for uid in user_ids}

    # Initialize each user with a unique label before iterative propagation.
    node_labels: dict[str, set[int]] = {}
    node_primary: dict[str, tuple[int, float]] = {}
    for i, uid in enumerate(user_ids):
        node_labels[uid] = {i}
        node_primary[uid] = (i, 1.0)

    # Update lower-importance nodes first so influential neighbors guide labels.
    sorted_users = sorted(user_ids, key=lambda u: ni_coeffs.get(u, 0.5))

    user_neighbors: dict[str, set[str]] = {}
    for uid, idx in user_to_idx.items():
        nbs = set()
        for nb in set(graph.successors(idx)) | set(graph.predecessors(idx)):
            if graph.nodes[nb].get("type") == "user":
                nb_uid = idx_to_user.get(nb)
                if nb_uid:
                    nbs.add(nb_uid)
        user_neighbors[uid] = nbs

    # Cache topic similarities once; the iterative loop reuses them repeatedly.
    similarities: dict[tuple, float] = {}
    topic_vecs = {uid: user_topics.get(uid, np.ones(LDA_TOPICS) / LDA_TOPICS)
                  for uid in user_ids}
    for uid in user_ids:
        for v in user_neighbors.get(uid, set()):
            if v not in topic_vecs:
                continue
            key = (uid, v) if uid < v else (v, uid)
            if key not in similarities:
                sim = cosine_similarity(topic_vecs[uid], topic_vecs[v])
                similarities[key] = sim

    def _get_sim(u, v):
        key = (u, v) if u < v else (v, u)
        return similarities.get(key, 0.0)

    for iteration in range(max_iterations):
        changed = False

        for uid in sorted_users:
            neighbors = user_neighbors.get(uid, set())
            if not neighbors:
                continue

            max_sim = 0.0
            sim_cache = {}
            for v in neighbors:
                sim_uv = _get_sim(uid, v)
                sim_cache[v] = sim_uv
                if sim_uv > max_sim:
                    max_sim = sim_uv

            if max_sim == 0:
                continue

            label_votes: dict[int, float] = defaultdict(float)
            total_weight = 0.0

            for v in neighbors:
                # Neighbor influence combines topic similarity and node importance.
                nni_v = ni_coeffs.get(v, 0.5) * math.sqrt(
                    sim_cache[v] / max_sim if max_sim > 0 else 0
                )
                primary_c, primary_b = node_primary.get(v, (-1, 0.0))
                if primary_c < 0:
                    continue
                weight = primary_b * nni_v
                label_votes[primary_c] += weight
                total_weight += weight

            if total_weight == 0:
                continue

            new_labels = {}
            for c, score in label_votes.items():
                prob = score / total_weight
                # Retain labels above the average vote share, allowing overlap.
                if prob >= 1.0 / max(1, len(label_votes)):
                    new_labels[c] = prob

            if not new_labels:
                continue

            best_c = max(new_labels, key=new_labels.get)
            best_b = new_labels[best_c]

            old_labels = node_labels.get(uid, set())
            if old_labels != set(new_labels.keys()):
                changed = True
                node_labels[uid] = set(new_labels.keys())
                node_primary[uid] = (best_c, best_b)

        if not changed:
            break

    log.info(f"communities · labels={len(set().union(*node_labels.values()))} iter={iteration + 1}")

    community_members: dict[int, set[str]] = defaultdict(set)
    for uid, labels in node_labels.items():
        for c in labels:
            community_members[c].add(uid)

    # Drop undersized communities and fall back to neighboring labels when needed.
    final_labels: dict[str, set[int]] = {}
    merged = set()
    for c, members in list(community_members.items()):
        if len(members) < MIN_COMMUNITY_SIZE:
            merged.add(c)

    if merged:
        for uid, labels in node_labels.items():
            filtered = labels - merged
            if not filtered:
                neighbor_comms = set()
                for v in user_neighbors.get(uid, set()):
                    if v in node_labels:
                        neighbor_comms |= node_labels[v] - merged
                final_labels[uid] = neighbor_comms if neighbor_comms else set()
            else:
                final_labels[uid] = filtered
    else:
        final_labels = node_labels

    return final_labels


def extract_community_features(
    community_labels: dict[str, set[int]], user_ids: list[str],
) -> np.ndarray:
    """Convert overlapping community assignments into a binary feature matrix."""
    all_comms = set()
    for labels in community_labels.values():
        all_comms.update(labels)

    comm_list = sorted(all_comms)
    comm_to_idx = {c: i for i, c in enumerate(comm_list)}
    n_comms = len(comm_list)

    features = np.zeros((len(user_ids), n_comms), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        if uid in community_labels:
            for c in community_labels[uid]:
                if c in comm_to_idx:
                    features[i, comm_to_idx[c]] = 1.0

    return features


def _compute_node2vec_embeddings(graph, user_to_idx: dict, n_users: int) -> np.ndarray:
    """Learn graph embeddings and place user vectors in user-id row order."""
    from node2vec import Node2Vec

    log.info(f"node2vec · dim={GRAPH_EMB_DIM} users={n_users}")
    t0 = time.time()
    n2v = Node2Vec(
        graph, dimensions=GRAPH_EMB_DIM, walk_length=20, num_walks=50,
        workers=1, quiet=True, seed=RANDOM_SEED,
    )
    with open(os.devnull, 'w') as devnull, redirect_stderr(devnull):
        model = n2v.fit(window=10, min_count=1, seed=RANDOM_SEED)

    embeddings = np.zeros((n_users, GRAPH_EMB_DIM), dtype=np.float32)
    for uid, idx in user_to_idx.items():
        key = str(idx)
        if key in model.wv:
            embeddings[idx, :] = model.wv[key]

    log.info(f"node2vec · {time.time() - t0:.1f}s")
    return embeddings


def recompute_rel_features(network, user_ids: list[str], topics: list[str]):
    """Recompute user_rel and entity_rel feature blocks."""
    user_rel_features = []
    entity_rel_features = []
    for uid in user_ids:
        user_feats = []
        entity_feats = []
        for topic in topics:
            user_feats.extend(network.get_user_rel_features(uid, topic))
            entity_feats.extend(network.get_entity_rel_features(uid, topic))
        user_rel_features.append(user_feats)
        entity_rel_features.append(entity_feats)
    return (np.array(user_rel_features, dtype=np.float32),
            np.array(entity_rel_features, dtype=np.float32))


def extract_all_features(
    data: dict, network, topics: list[str], dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build all feature blocks, heuristic labels, and the saved NPZ artifact.

    LDA, community detection, and node2vec embeddings are computed on the full
    corpus before train/test split (transductive setting). Only PageRank-based
    features are recomputed per CV fold in ml_knn.py.
    """
    log.info(f"▸ {dataset_name}")
    t0 = time.time()

    user_ids = list(data.get("user_ids", []))
    seed_posts = data.get("seed_posts", {})
    seed_users = data.get("seed_users", {})

    user_texts = build_user_texts(seed_posts)
    user_topics = run_lda(user_texts)

    ni_coeffs = compute_ni_coefficient(network.graph, network.user_to_idx)

    community_labels = overlapping_community_detection(
        network.graph, network.user_to_idx, network.idx_to_user,
        user_topics, ni_coeffs,
    )

    comm_features = extract_community_features(community_labels, user_ids)

    user_rel_features = []
    entity_rel_features = []
    for uid in user_ids:
        user_feats = []
        entity_feats = []
        for topic in topics:
            user_feats.extend(network.get_user_rel_features(uid, topic))
            entity_feats.extend(network.get_entity_rel_features(uid, topic))
        user_rel_features.append(user_feats)
        entity_rel_features.append(entity_feats)

    user_rel_arr = np.array(user_rel_features, dtype=np.float32)
    entity_rel_arr = np.array(entity_rel_features, dtype=np.float32)

    graph_emb = _compute_node2vec_embeddings(network.graph, network.user_to_idx, len(user_ids))

    # Feature block order is consumed by compare_baselines.py when slicing X.
    lda_dim = min(LDA_TOPICS, next(iter(user_topics.values())).shape[0] if user_topics else LDA_TOPICS)
    lda_arr = np.zeros((len(user_ids), lda_dim), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        if uid in user_topics:
            lda_arr[i, :lda_dim] = user_topics[uid][:lda_dim]

    profiles = data.get("user_profiles", {})
    prof_arr = np.zeros((len(user_ids), PROFILE_DIM), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        p = profiles.get(uid, {})
        if p:
            prof_arr[i, 0] = np.log1p(p.get("followers_count", 0))
            prof_arr[i, 1] = np.log1p(p.get("following_count", 0))
            prof_arr[i, 2] = np.log1p(p.get("statuses_count", 0))
            prof_arr[i, 3] = np.log1p(p.get("following_count", 0) / max(1, p.get("followers_count", 1)))

    X = np.hstack([user_rel_arr, entity_rel_arr, comm_features, lda_arr, prof_arr, graph_emb])
    log.info(f"features · {X.shape[1]} user_rel={user_rel_arr.shape[1]} entity_rel={entity_rel_arr.shape[1]} comm={comm_features.shape[1]} lda={lda_arr.shape[1]} prof={prof_arr.shape[1]} emb={graph_emb.shape[1]}")

    # Labels are weak supervision generated from seed membership and observed hashtags.
    # They are not manually verified ground-truth labels.
    # Hard labels remain conservative, while label_scores retain confidence-weighted evidence for ML-KNN training.
    topic_to_idx = {t: i for i, t in enumerate(topics)}
    n_users = len(user_ids)
    y = np.zeros((n_users, len(topics)), dtype=np.int32)
    label_scores = np.zeros((n_users, len(topics)), dtype=np.float32)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}

    for i, uid in enumerate(user_ids):
        user_tags = set()
        if uid in seed_posts:
            for post in seed_posts[uid]:
                for tag in post.get("tags", []):
                    user_tags.add(tag["name"].lower())

        for topic in topics:
            ti = topic_to_idx[topic]
            score = 0

            # Seed membership and explicit topic tags provide stronger evidence.
            for acc in seed_users.get(topic, []):
                if acc.get("id", acc) == uid:
                    score += 3
                    break
            if topic in user_tags or any(
                t in user_tags for t in [topic + "s", topic + "news", topic + "discussion"]
            ):
                score += 2
            if score >= 2:
                y[i, ti] = 1
            label_scores[i, ti] = min(1.0, score / 5.0)

    # Community-level seed hits raise confidence without changing hard labels.
    comm_members = defaultdict(list)
    comm_seed_hits = {}
    for uid, labels in community_labels.items():
        pos = user_to_pos.get(uid)
        if pos is None:
            continue
        for c in labels:
            comm_members[c].append(pos)
            for topic in topics:
                ti = topic_to_idx[topic]
                for acc in seed_users.get(topic, []):
                    if acc.get("id", acc) == uid:
                        comm_seed_hits[(c, ti)] = comm_seed_hits.get((c, ti), 0) + 1

    for (c, ti), seed_count in comm_seed_hits.items():
        members = comm_members[c]
        if len(members) < MIN_COMMUNITY_SIZE:
            continue
        strength = seed_count / max(3, len(members))
        for pos in members:
            current = label_scores[pos, ti]
            bonus = strength * 0.6
            if bonus > current:
                label_scores[pos, ti] = bonus

    labeled_count = (y.sum(axis=1) > 0).sum()
    log.info(f"labeled · {labeled_count}/{len(user_ids)}")
    label_dist = " ".join(f"{t}={c}" for t, c in zip(topics, y.sum(axis=0)))
    log.info(f"labels · {label_dist}")

    features_path = os.path.join(DATA_DIR, f"{dataset_name}_features.npz")
    np.savez(features_path, X=X, y=y, label_scores=label_scores,
             user_ids=np.array(user_ids), topics=topics, lda_dim=lda_dim,
             n_topics=len(topics))
    log.info(f"file · {features_path}")
    log.info(f"total · {time.time() - t0:.1f}s")

    return X, y, user_ids


if __name__ == "__main__":
    t0 = time.time()
    log.header()
    log.info("args · all")
    for ds, topics in [("dataset1", DATASET1_TOPICS), ("dataset2", DATASET2_TOPICS)]:
        data_path = os.path.join(DATA_DIR, f"{ds}.pkl")
        net_path = os.path.join(DATA_DIR, f"{ds}_network.pkl")
        if not os.path.exists(data_path) or not os.path.exists(net_path):
            log.skip(f"{ds} data not found")
            continue

        with open(data_path, "rb") as f:
            data = pickle.load(f)
        with open(net_path, "rb") as f:
            network = pickle.load(f)

        extract_all_features(data, network, topics, ds)
    log.ok(f"features complete · {time.time() - t0:.1f}s")
    log.blank()
