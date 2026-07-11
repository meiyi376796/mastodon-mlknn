"""Construct heterogeneous user/entity networks from collected Mastodon data."""

import math
import os
import pickle
import time
from collections import defaultdict
from urllib.parse import urlparse

import networkx as nx
import numpy as np

from config import DATA_DIR, DATASETS, USER_REL_DIM, ENTITY_REL_DIM
from logger import Logger

log = Logger("network")


def _extract_entities(statuses: list[dict]) -> list[str]:
    """Extract hashtag and linked-domain entities from a user's statuses."""
    entities = set()
    for s in statuses:
        for tag in s.get("tags", []):
            entities.add(f"tag:{tag['name'].lower()}")
        if s.get("card") and s["card"].get("url"):
            domain = urlparse(s["card"]["url"]).netloc
            if domain:
                entities.add(f"domain:{domain}")
    return list(entities)


class HeterogeneousNetwork:
    """NetworkX-backed graph containing user-user and user-entity relationships."""

    def __init__(self, data: dict, topics: list[str]):
        self.data = data
        self.topics = topics
        self.user_ids = list(data.get("user_ids", []))
        self.relationships = data.get("relationships", {})
        self.seed_posts = data.get("seed_posts", {})

        self.graph = nx.MultiDiGraph()
        self.user_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}

        self.rs_scores: dict[str, dict[str, float]] = {}
        self.train_user_ids: set[str] | None = None
        self._domain_entities_cache: dict[str, frozenset] = {}

        self._build()

    def _build(self):
        # Add users first to keep graph embedding rows aligned with user_ids.
        for i, uid in enumerate(self.user_ids):
            self.user_to_idx[uid] = i
            self.idx_to_user[i] = uid
            self.graph.add_node(i, type="user")

        # Collect content entities before assigning graph indices.
        all_entities = set()
        user_entities: dict[str, set[str]] = defaultdict(set)
        for uid, posts in self.seed_posts.items():
            entities = _extract_entities(posts)
            user_entities[uid] = set(entities)
            all_entities.update(entities)

        entity_offset = len(self.user_ids)
        for i, ent in enumerate(all_entities):
            self.entity_to_idx[ent] = entity_offset + i
            self.idx_to_entity[entity_offset + i] = ent
            self.graph.add_node(entity_offset + i, type="entity")

        # Keep social edges inside the observed user set.
        for uid, rels in self.relationships.items():
            if uid not in self.user_to_idx:
                continue
            ui = self.user_to_idx[uid]
            for method in ["follows", "boosts", "mentions", "replies"]:
                for target in rels.get(method, []):
                    if target in self.user_to_idx:
                        ti = self.user_to_idx[target]
                        self.graph.add_edge(ui, ti, relation=method)

        # Content edges connect users through shared tags or domains.
        for uid, entities in user_entities.items():
            if uid not in self.user_to_idx:
                continue
            ui = self.user_to_idx[uid]
            for ent in entities:
                if ent in self.entity_to_idx:
                    ei = self.entity_to_idx[ent]
                    self.graph.add_edge(ui, ei, relation="uses")

    def set_train_fold(self, train_ids: set[str] | None):
        """Restrict seed-dependent computations to training users."""
        self.train_user_ids = train_ids
        self.rs_scores = {}

    def compute_rs_scores(self, max_iter: int = 100, alpha: float = 0.85):
        """Compute topic-specific relevance scores using personalized PageRank."""
        seed_users_by_topic = self.data.get("seed_users", {})

        for topic in self.topics:
            personalization = {}
            seed_list = seed_users_by_topic.get(topic, [])
            if not seed_list:
                log.warn(f"PageRank · {topic} · no seed users skipping")
                continue

            # Restrict seeds to training users to prevent label leakage across folds.
            if self.train_user_ids is not None:
                seed_list = [acc for acc in seed_list
                             if acc.get("id", acc) in self.train_user_ids]

            if not seed_list:
                log.warn(f"PageRank · {topic} · no training seeds skipping")
                continue

            weight = 1.0 / len(seed_list)
            for acc in seed_list:
                uid = acc.get("id", acc)
                if uid in self.user_to_idx:
                    personalization[self.user_to_idx[uid]] = weight

            if not personalization:
                log.warn(f"PageRank · {topic} · no valid seed indices skipping")
                continue

            start_time = time.time()
            try:
                pr = nx.pagerank(
                    self.graph, alpha=alpha,
                    personalization=personalization,
                    max_iter=max_iter, tol=1e-6
                )
            except Exception as e:
                log.warn(f"PageRank · {topic} · failed ({e})")
                continue

            self.rs_scores[topic] = {}
            for uid in self.user_ids:
                if uid in self.user_to_idx:
                    idx = self.user_to_idx[uid]
                    self.rs_scores[topic][uid] = pr.get(idx, 0.0)
            elapsed = time.time() - start_time
            log.info(f"PageRank · {topic} · {elapsed:.1f}s")

    def get_user_rel_features(self, uid: str, topic: str) -> list[float]:
        """Summarize neighboring users' topic relevance scores for one user."""
        if uid not in self.user_to_idx:
            return [0.0] * USER_REL_DIM

        ui = self.user_to_idx[uid]
        rs_topic = self.rs_scores.get(topic, {})

        all_scores = []
        for nb in set(self.graph.successors(ui)) | set(self.graph.predecessors(ui)):
            if not isinstance(nb, int):
                continue
            if self.graph.nodes[nb].get("type") != "user":
                continue
            nb_uid: str | None = self.idx_to_user.get(nb)
            if nb_uid and nb_uid in rs_topic:
                all_scores.append(rs_topic[nb_uid])

        if not all_scores:
            return [0.0] * USER_REL_DIM

        scores = np.array(all_scores, dtype=np.float64)
        return [
            float(np.mean(scores)),
            float(np.max(scores)),
            float(np.min(scores)),
            float(np.std(scores)),
            float(np.median(scores)),
            float(len(scores)),
            float(np.sum(scores)),
            float(np.var(scores)),
        ]

    def _domain_entities_of(self, uid: str) -> frozenset:
        """Return the domain entities of a user's posts, computed once and cached."""
        if uid not in self._domain_entities_cache:
            entities = _extract_entities(self.seed_posts.get(uid, []))
            self._domain_entities_cache[uid] = frozenset(
                e for e in entities if e.startswith("domain:")
            )
        return self._domain_entities_cache[uid]

    def get_entity_rel_features(self, uid: str, topic: str) -> list[float]:
        """Measure domain-entity overlap between a user's entities and topic seed entities."""
        if uid not in self.user_to_idx:
            return [0.0] * ENTITY_REL_DIM

        ui = self.user_to_idx[uid]
        entity_neighbors = []
        for nb in self.graph.successors(ui):
            if not isinstance(nb, int):
                continue
            if self.graph.nodes[nb].get("type") == "entity":
                ent_name = self.idx_to_entity.get(nb, "")
                if ent_name.startswith("domain:"):
                    entity_neighbors.append(ent_name)

        if not entity_neighbors:
            return [0.0] * ENTITY_REL_DIM

        topic_entities = set()
        for acc in self.data.get("seed_users", {}).get(topic, []):
            uid_s = acc.get("id", acc)
            if uid_s == uid:
                continue
            if self.train_user_ids is not None and uid_s not in self.train_user_ids:
                continue
            if uid_s in self.seed_posts:
                topic_entities |= self._domain_entities_of(uid_s)

        overlap = len(set(entity_neighbors) & topic_entities)
        ratio = overlap / max(1, len(entity_neighbors))

        features = [
            math.log1p(len(entity_neighbors)),
            math.log1p(overlap),
            ratio,
            math.log1p(len(entity_neighbors) - overlap),
            math.log1p(len(topic_entities)),
            overlap / max(1, len(topic_entities)),
        ]
        return features


def build_network(data: dict, topics: list[str], dataset_name: str) -> HeterogeneousNetwork:
    """Build, score, and persist a heterogeneous network for one dataset."""
    log.info(f"▸ {dataset_name}")
    start_time = time.time()
    net = HeterogeneousNetwork(data, topics)
    log.info(f"nodes · {net.graph.number_of_nodes()}")
    log.info(f"edges · {net.graph.number_of_edges()}")
    net.compute_rs_scores()

    path = os.path.join(DATA_DIR, f"{dataset_name}_network.pkl")
    with open(path, "wb") as f:
        pickle.dump(net, f)
    log.info(f"file · {path}")
    log.info(f"total · {time.time() - start_time:.1f}s")

    return net


def _build_one_dataset(dataset_name: str, dataset_topics: list[str]) -> HeterogeneousNetwork | None:
    """Load a pickled dataset and build its heterogeneous network."""
    data_path = os.path.join(DATA_DIR, f"{dataset_name}.pkl")
    if not os.path.exists(data_path):
        log.skip(f"{data_path} not found")
        return None
    with open(data_path, "rb") as f:
        loaded_data = pickle.load(f)
    return build_network(loaded_data, dataset_topics, dataset_name)


if __name__ == "__main__":
    total_start = time.time()
    log.header()
    log.info("args · all")
    for ds, topics in DATASETS:
        _build_one_dataset(ds, topics)
    log.ok(f"network complete · {time.time() - total_start:.1f}s")
    log.blank()
