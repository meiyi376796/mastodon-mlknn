"""Fetch Mastodon data used to build the multi-label classification datasets."""

import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

import requests

from config import (
    DATASETS,
    SEED_USERS_PER_TOPIC, POSTS_PER_SEED,
    DATA_DIR, INSTANCES, RANDOM_SEED,
)
from logger import Logger

log = Logger("fetch")

os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "mastodon-mlknn/1.0"}
REQUEST_DELAY = 0.3
REQUEST_JITTER = 0.3


def _normalize_uid(account: dict, instance: str) -> str:
    """Build a globally unique user identifier from a Mastodon account dict."""
    acct = account.get("acct", "")
    domain = urlparse(instance).netloc
    if acct:
        if '@' in acct:
            return acct
        return f"{acct}@{domain}"
    return f"{domain}#{account.get('id', '')}"


def _build_profile(uid: str, account: dict, instance: str = "") -> dict:
    """Keep only stable account fields needed by downstream feature extraction."""
    return {
        "id": uid,
        "raw_id": account.get("id", ""),
        "instance": instance,
        "username": account.get("username", ""),
        "acct": account.get("acct", ""),
        "display_name": account.get("display_name", ""),
        "url": account.get("url", ""),
        "followers_count": account.get("followers_count", 0),
        "following_count": account.get("following_count", 0),
        "statuses_count": account.get("statuses_count", 0),
    }


def _paginated_get(url: str, limit: int, page_size: int = 40,
                   extra_params: dict | None = None) -> list[dict]:
    """Read Mastodon paginated endpoints until the requested limit is reached."""
    all_items = []
    max_id: str | None = None
    while len(all_items) < limit:
        params: dict[str, Any] = {"limit": min(page_size, limit - len(all_items))}
        if extra_params:
            params.update(extra_params)
        if max_id:
            params["max_id"] = max_id
        results = _get(url, params)
        if not results or not isinstance(results, list):
            break
        all_items.extend(results)
        if len(results) < page_size:
            break
        # Mastodon uses max_id for backwards pagination through timeline results.
        max_id = results[-1]["id"]
        time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))
    return all_items[:limit]


def _get(url: str, params: dict | None = None, max_retries: int = 5) -> Any:
    """Perform HTTP GET with bounded retry/backoff handling for public API collection."""
    last_error = None
    last_http_status = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=(5, 30))
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = min(300, (2 ** attempt) * 5)
                wait += random.uniform(0, 5)
                log.warn(f"rate limited · wait {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            elif r.status_code in (404, 410):
                return []
            else:
                last_http_status = r.status_code
                log.warn(f"HTTP {r.status_code} (attempt {attempt + 1}/{max_retries})")
                time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            last_error = "connection"
            if attempt >= 2:
                log.warn(f"connection failed (attempt {attempt + 1}/{max_retries})")
                return []
            log.warn(f"connection failed (attempt {attempt + 1}/{max_retries})")
            time.sleep(REQUEST_DELAY * 2)
        except Exception as e:
            last_error = e
            log.warn(f"request failed (attempt {attempt + 1}/{max_retries})")
            time.sleep(REQUEST_DELAY * 2)
    if last_error:
        log.error(f"request failed after {max_retries} attempts")
    elif last_http_status:
        log.error(f"HTTP {last_http_status} persisted after {max_retries} attempts")
    return []


def get_tag_timeline(instance: str, tag: str, limit: int = 200) -> list[dict]:
    url = f"{instance}/api/v1/timelines/tag/{tag}"
    return _paginated_get(url, limit, page_size=40)


def get_account_statuses(instance: str, account_id: str, limit: int = 80) -> list[dict]:
    url = f"{instance}/api/v1/accounts/{account_id}/statuses"
    return _paginated_get(url, limit, page_size=40,
                          extra_params={"exclude_replies": False, "exclude_reblogs": False})


def get_account_by_acct(instance: str, acct: str) -> dict[str, Any] | list[Any]:
    """Look up a remote account on a given instance via WebFinger/acct."""
    url = f"{instance}/api/v1/accounts/lookup"
    return _get(url, {"acct": acct}, max_retries=2)


def get_following(instance: str, account_id: str, limit: int = 80) -> list[dict]:
    url = f"{instance}/api/v1/accounts/{account_id}/following"
    return _paginated_get(url, limit, page_size=80)


def _resolve_account(profile: dict) -> tuple[str, str]:
    """Return (instance, raw_id) for an account, falling back to working instances."""
    inst = profile.get("instance", "")
    raw_id = profile.get("raw_id", "")
    acct = profile.get("acct", "")
    if inst and raw_id:
        return inst, raw_id
    for fallback in INSTANCES:
        looked_up = get_account_by_acct(fallback, acct)
        if isinstance(looked_up, dict) and looked_up.get("id"):
            profile["instance"] = fallback
            profile["raw_id"] = looked_up["id"]
            return fallback, looked_up["id"]
    return INSTANCES[0], raw_id


def _fetch_with_fallback(fetch_fn, profile: dict, *args, **kwargs):
    """Call fetch_fn(instance, raw_id, ...); on failure, resolve on other instances."""
    inst, raw_id = _resolve_account(profile)
    result = fetch_fn(inst, raw_id, *args, **kwargs)
    if result:
        return result
    acct = profile.get("acct", "")
    if not acct:
        return []
    for fallback in INSTANCES:
        if fallback == inst:
            continue
        looked_up = get_account_by_acct(fallback, acct)
        if isinstance(looked_up, dict) and looked_up.get("id"):
            profile["instance"] = fallback
            profile["raw_id"] = looked_up["id"]
            result = fetch_fn(fallback, looked_up["id"], *args, **kwargs)
            if result:
                return result
    return []


def _save_full(dataset_name, topics, seed_users, seed_posts, all_user_ids,
               relationships, user_profiles):
    """Save the full dataset snapshot to .pkl for crash recovery."""
    path = _final_data_path(dataset_name)
    data = {
        "dataset_name": dataset_name,
        "topics": topics,
        "seed_users": dict(seed_users),
        "seed_posts": seed_posts,
        "user_ids": sorted(all_user_ids),
        "relationships": dict(relationships),
        "user_profiles": user_profiles,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _checkpoint_path(dataset_name: str) -> str:
    """Path for the progress-only checkpoint file."""
    return os.path.join(DATA_DIR, f"{dataset_name}_progress.pkl")


def _save_checkpoint(dataset_name, processed_seeds, processed_secondary,
                     topics_done, seed_phase_done, secondary_sample):
    """Save progress markers to a tiny checkpoint file."""
    path = _checkpoint_path(dataset_name)
    data = {
        "_processed_seeds": list(processed_seeds),
        "_processed_secondary": list(processed_secondary),
        "_topics_done": list(topics_done),
        "_seed_phase_done": seed_phase_done,
        "_secondary_sample": secondary_sample,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_checkpoint(dataset_name: str) -> dict | None:
    """Load progress markers; merge with full data if .pkl exists."""
    path = _checkpoint_path(dataset_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            progress = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        log.warn(f"checkpoint corrupted · discarding")
        os.remove(path)
        return None
    data_path = _final_data_path(dataset_name)
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = {}
    data.update(progress)
    return data


def _final_data_path(dataset_name: str) -> str:
    """Path for the final dataset pickle."""
    return os.path.join(DATA_DIR, f"{dataset_name}.pkl")


def collect_data(topics: list[str], dataset_name: str) -> dict:
    """Collect seed users, posts, and relationship edges for a topic set with checkpoint resume."""
    log.info(f"▸ {dataset_name}")
    start_time = time.time()

    ckpt = _load_checkpoint(dataset_name)
    if ckpt:
        seed_users = defaultdict(list, ckpt.get("seed_users", {}))
        seed_posts = ckpt.get("seed_posts", {})
        all_user_ids = set(ckpt.get("user_ids", []))
        relationships = defaultdict(
            lambda: {"follows": [], "mentions": [], "boosts": [], "replies": []},
            {k: v for k, v in ckpt.get("relationships", {}).items()}
        )
        user_profiles = ckpt.get("user_profiles", {})
        processed_seeds = set(ckpt.get("_processed_seeds", []))
        processed_secondary = set(ckpt.get("_processed_secondary", []))
        topics_done = set(ckpt.get("_topics_done", []))
        seed_phase_done = ckpt.get("_seed_phase_done", False)
        secondary_sample = ckpt.get("_secondary_sample", [])
        log.info(f"resumed checkpoint · seeds={len(processed_seeds)} secondary={len(processed_secondary)}")
    else:
        seed_users = defaultdict(list)
        seed_posts = {}
        all_user_ids = set()
        relationships = defaultdict(
            lambda: {"follows": [], "mentions": [], "boosts": [], "replies": []}
        )
        user_profiles = {}
        processed_seeds = set()
        processed_secondary = set()
        topics_done = set()
        seed_phase_done = False
        secondary_sample = []

    for topic in topics:
        if topic in topics_done:
            log.info(f"{topic} · skipped")
            continue
        topic_candidates = []
        seen_in_topic = set()

        for instance in INSTANCES:
            posts = get_tag_timeline(instance, topic, limit=200)
            for post in posts:
                uid = _normalize_uid(post["account"], instance)
                if uid not in user_profiles:
                    user_profiles[uid] = _build_profile(uid, post["account"], instance)

                if uid not in seen_in_topic:
                    seen_in_topic.add(uid)
                    topic_candidates.append(uid)

                all_user_ids.add(uid)
                if uid not in seed_posts:
                    seed_posts[uid] = []
                seed_posts[uid].append(post)

            time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))

        topic_candidates.sort(key=lambda u: user_profiles[u]["followers_count"], reverse=True)
        log.info(f"{topic} · {len(topic_candidates)} candidates")
        for uid in topic_candidates[:SEED_USERS_PER_TOPIC]:
            seed_users[topic].append(user_profiles[uid])

        log.info(f"{topic} · {len(seed_users[topic])} seeds")
        if len(topic_candidates) > 0:
            topics_done.add(topic)
        else:
            log.warn(f"{topic} · no candidates, will retry")
        _save_checkpoint(dataset_name,
            processed_seeds, processed_secondary,
            topics_done, seed_phase_done, secondary_sample)
        _save_full(dataset_name, topics, seed_users, seed_posts, all_user_ids,
                   relationships, user_profiles)

    all_seed_ids = set()
    ordered_seed_ids = []
    for topic_seeds in seed_users.values():
        for acc in topic_seeds:
            uid = acc["id"]
            if uid not in all_seed_ids:
                all_seed_ids.add(uid)
                ordered_seed_ids.append(uid)

    if not seed_phase_done:
        pending_seeds = [uid for uid in ordered_seed_ids if uid not in processed_seeds]
        processed = len(processed_seeds)
        for uid in pending_seeds:
            inst = user_profiles[uid].get("instance", INSTANCES[0])
            processed += 1
            if processed % 30 == 0 or processed == len(ordered_seed_ids):
                log.info(f"seeds · {processed}/{len(ordered_seed_ids)}")

            # deduplicate relationships on resume
            relationships[uid] = {"follows": [], "mentions": [], "boosts": [], "replies": []}

            posts_before = len(seed_posts.get(uid, []))
            if posts_before < 20:
                statuses = _fetch_with_fallback(get_account_statuses, user_profiles[uid], POSTS_PER_SEED)
                if uid not in seed_posts:
                    seed_posts[uid] = []
                existing_ids = {s["id"] for s in seed_posts[uid]}
                for s in statuses:
                    if s["id"] not in existing_ids:
                        seed_posts[uid].append(s)

            for s in seed_posts.get(uid, []):
                for m in s.get("mentions", []):
                    mid = _normalize_uid(m, inst)
                    all_user_ids.add(mid)
                    relationships[uid]["mentions"].append(mid)
                if s.get("reblog"):
                    boost_uid = _normalize_uid(s["reblog"]["account"], inst)
                    all_user_ids.add(boost_uid)
                    relationships[uid]["boosts"].append(boost_uid)
                if s.get("in_reply_to_account_id"):
                    domain = urlparse(inst).netloc
                    reply_acct = s.get("in_reply_to_account_acct", "")
                    if reply_acct:
                        if '@' in reply_acct:
                            reply_uid = reply_acct
                        else:
                            reply_uid = f"{reply_acct}@{domain}"
                    else:
                        reply_uid = f"{domain}#{s['in_reply_to_account_id']}"
                    all_user_ids.add(reply_uid)
                    relationships[uid]["replies"].append(reply_uid)

            following = _fetch_with_fallback(get_following, user_profiles[uid], 80)
            for f in following:
                fid = _normalize_uid(f, inst)
                all_user_ids.add(fid)
                relationships[uid]["follows"].append(fid)
                if fid not in user_profiles:
                    user_profiles[fid] = _build_profile(fid, f, inst)
                if fid not in seed_posts:
                    seed_posts[fid] = []

            if posts_before >= 20 or len(seed_posts.get(uid, [])) > posts_before:
                processed_seeds.add(uid)
            else:
                log.warn(f"seeds · {uid} empty fetch, will retry")

            # checkpoint every 30 seeds
            if processed % 30 == 0:
                _save_checkpoint(dataset_name,
                    processed_seeds, processed_secondary,
                    topics_done, seed_phase_done, secondary_sample)
                _save_full(dataset_name, topics, seed_users, seed_posts, all_user_ids,
                           relationships, user_profiles)

            time.sleep(REQUEST_DELAY + random.uniform(0, REQUEST_JITTER))

        seed_phase_done = (len(processed_seeds) >= len(ordered_seed_ids))
        _save_checkpoint(dataset_name,
            processed_seeds, processed_secondary,
            topics_done, seed_phase_done, secondary_sample)
        _save_full(dataset_name, topics, seed_users, seed_posts, all_user_ids,
                   relationships, user_profiles)

    random.seed(RANDOM_SEED)
    if not secondary_sample:
        secondary_users = sorted(all_user_ids - all_seed_ids)
        secondary_sample = random.sample(secondary_users, min(200, len(secondary_users)))
    if not secondary_sample:
        log.info("secondary · none")
    else:
        pending_secondary = [uid for uid in secondary_sample if uid not in processed_secondary]
        sec_processed = len(processed_secondary)
        for uid in pending_secondary:
            sec_processed += 1
            if sec_processed % 50 == 0 or sec_processed == len(secondary_sample):
                log.info(f"secondary · {sec_processed}/{len(secondary_sample)}")

            if uid not in user_profiles:
                processed_secondary.add(uid)
                continue
            inst = user_profiles[uid].get("instance", INSTANCES[0])
            statuses = _fetch_with_fallback(get_account_statuses, user_profiles[uid], 30)
            if uid not in seed_posts:
                seed_posts[uid] = []
            posts_before = len(seed_posts[uid])
            existing_ids = {s["id"] for s in seed_posts[uid]}
            for s in statuses:
                if s["id"] not in existing_ids:
                    seed_posts[uid].append(s)
                for m in s.get("mentions", []):
                    mid = _normalize_uid(m, inst)
                    all_user_ids.add(mid)
                    relationships[uid]["mentions"].append(mid)
            if not statuses and posts_before == 0:
                log.warn(f"secondary · {uid} empty fetch, will retry")
            else:
                processed_secondary.add(uid)
            time.sleep(REQUEST_DELAY / 2 + random.uniform(0, REQUEST_JITTER / 2))

            # checkpoint every 50 secondary users
            if sec_processed % 50 == 0:
                _save_checkpoint(dataset_name,
                    processed_seeds, processed_secondary,
                    topics_done, seed_phase_done, secondary_sample)

    elapsed = time.time() - start_time
    n_posts = len([u for u, p in seed_posts.items() if p])
    n_relations = sum(1 for r in relationships.values() if any(r.values()))
    log.info(f"users · {len(all_user_ids)}")
    log.info(f"posts · {n_posts}")
    log.info(f"relations · {n_relations}")
    log.info(f"total · {elapsed:.1f}s")

    relationships_serializable = {}
    for uid, rels in relationships.items():
        relationships_serializable[uid] = {
            k: list(set(v)) for k, v in rels.items()
        }

    data: dict[str, Any] = {
        "dataset_name": dataset_name,
        "topics": topics,
        "seed_users": dict(seed_users),
        "seed_posts": seed_posts,
        "user_ids": sorted(all_user_ids),
        "relationships": relationships_serializable,
        "user_profiles": user_profiles,
        "_processed_seeds": list(processed_seeds),
        "_processed_secondary": list(processed_secondary),
        "_topics_done": list(topics_done),
        "_seed_phase_done": seed_phase_done,
        "_secondary_sample": secondary_sample,
    }

    path = _final_data_path(dataset_name)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log.info(f"file · {path}")

    return data


if __name__ == "__main__":
    total_start = time.time()
    log.header()
    log.info("args · all")
    for ds, topics in DATASETS:
        collect_data(topics, ds)
    log.ok(f"fetch complete · {time.time() - total_start:.1f}s")
    log.blank()
