"""Pipeline entry point for the Mastodon multi-label classification workflow."""

import os
import pickle
import sys

from config import (
    DATA_DIR, DATASETS, METRIC_LABELS, METRIC_ORDER,
    K_NEIGHBORS, SMOOTHING_FACTOR, LDA_TOPICS, GRAPH_EMB_DIM, PROFILE_DIM,
)
from logger import Logger

log = Logger("pipeline")


def step1_fetch_data():
    from data_fetcher import collect_data
    for ds, topics in DATASETS:
        collect_data(topics, ds)


def _run_for_dataset(callback, load_network=False):
    """Load data (and network if requested) for each dataset and invoke callback."""
    for ds, topics in DATASETS:
        data_path = os.path.join(DATA_DIR, f"{ds}.pkl")
        required_paths = [data_path]
        network = None
        net_path = ""
        if load_network:
            net_path = os.path.join(DATA_DIR, f"{ds}_network.pkl")
            required_paths.append(net_path)
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            log.skip(f"{', '.join(missing)} not found")
            continue
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if load_network:
            with open(net_path, "rb") as f:
                network = pickle.load(f)
        callback(data, network, topics, ds)


def step2_build_network():
    from network_builder import build_network

    def _on_dataset(data, _network, topics, ds):
        build_network(data, topics, ds)

    _run_for_dataset(_on_dataset, load_network=False)


def step3_extract_features():
    from feature_extractor import extract_all_features

    def _on_dataset(data, network, topics, ds):
        extract_all_features(data, network, topics, ds)

    _run_for_dataset(_on_dataset, load_network=True)


def step4_train_evaluate():
    from ml_knn import run_experiment, run_nested_cv
    results = {}
    for ds, _ in DATASETS:
        feat_path = os.path.join(DATA_DIR, f"{ds}_features.npz")
        if not os.path.exists(feat_path):
            log.skip(f"{feat_path} not found")
            continue
        run_nested_cv(ds)
        metrics = run_experiment(ds)
        if metrics is None:
            log.warn(f"{ds} · no results (insufficient labeled users)")
        else:
            results[ds] = metrics

    if not results:
        log.warn("TRAIN · no features available")
        return results

    log.header("RESULTS")
    log.info(f"{'Metric':20s} {'Dataset1':>10s} {'Dataset2':>10s}")

    for metric in METRIC_ORDER:
        v1 = results.get("dataset1", {}).get(metric, 0)
        v2 = results.get("dataset2", {}).get(metric, 0)
        log.info(f"{METRIC_LABELS[metric]:20s} {v1:10.4f} {v2:10.4f}")

    return results


def main():
    """Run every stage by default, or selected stages from CLI arguments."""
    import time
    start_time = time.time()

    log.header()
    log.info(f"args · {' '.join(sys.argv[1:]) if sys.argv[1:] else 'all'}")
    log.info(f"cfg · k={K_NEIGHBORS} s={SMOOTHING_FACTOR} lda={LDA_TOPICS} emb={GRAPH_EMB_DIM} prof={PROFILE_DIM}")

    steps = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    valid_steps = {"fetch", "network", "features", "train", "all"}
    unknown = set(steps) - valid_steps
    if unknown:
        log.error(f"unknown step(s): {' '.join(sorted(unknown))}")
        sys.exit(1)
    run_all = "all" in steps

    if run_all or "fetch" in steps:
        log.header("FETCH")
        log.info("starting …")
        step1_fetch_data()
    else:
        log.info("FETCH · skipped")

    if run_all or "network" in steps:
        log.header("NETWORK")
        log.info("starting …")
        step2_build_network()
    else:
        log.info("NETWORK · skipped")

    if run_all or "features" in steps:
        log.header("FEATURES")
        log.info("starting …")
        step3_extract_features()
    else:
        log.info("FEATURES · skipped")

    trained = False
    if run_all or "train" in steps:
        log.header("TRAIN")
        log.info("starting …")
        results = step4_train_evaluate()
        trained = bool(results)
    else:
        log.info("TRAIN · skipped")

    elapsed = time.time() - start_time
    if (run_all or "train" in steps) and not trained:
        log.warn("pipeline · no results produced")
        sys.exit(1)
    else:
        log.ok(f"pipeline complete · {elapsed:.1f}s")
    log.blank()


if __name__ == "__main__":
    main()
