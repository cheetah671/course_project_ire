"""Step 4: train/dev/test split BY QUERY, grouped so that no two splits share supporting documents.

Leakage risk: HotpotQA questions are built from linked article pairs; many questions share a supporting
article (and NQ/TriviaQA questions can share answer-bearing articles). If such questions land in both
train and test, the router can memorise "questions about article X succeed with config Y", and Router B's
retrieval features (which see those same passages) would look better than they are.

Procedure:
 1. Link two queries if they share a gold title, unless the title is a "hub" (gold title of more than
    `hub_title_max_queries` queries; linking through hubs would collapse everything into one component).
 2. Link exact duplicates after answer-style normalisation of the question text, and, if question
    embeddings are supplied, pairs with cosine >= near_dup_cosine.
 3. Connected components (union-find) are the units of assignment. Components are visited in a seeded
    random order (largest first) and each is put into the split whose per-dataset fill ratio stays lowest,
    so each dataset ends close to 60/20/20.
 4. Report: residual overlap (test queries sharing ANY gold title, hubs included, with train), the
    hub titles, and the component-size distribution. The report goes in the appendix.

Output: {data_root}/splits.parquet (qid, split, component) and {results}/leakage_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment
from ragroute.eval.metrics import normalize_answer

SPLITS = ("train", "dev", "test")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def grouped_split(meta: pd.DataFrame, ratios: dict, seed: int, hub_max: int,
                  embeddings: np.ndarray | None = None, near_dup: float = 0.95, max_component_frac: float = 0.05):
    meta = meta.reset_index(drop=True)
    n = len(meta)
    uf = UnionFind(n)
    title_users = defaultdict(list)
    for i, titles in enumerate(meta["gold_titles"]):
        for t in set(titles):
            title_users[t].append(i)
    hubs = {t for t, users in title_users.items() if len(users) > hub_max}
    for t, users in title_users.items():
        if t in hubs:
            continue
        for j in users[1:]:
            uf.union(users[0], j)
    norm_q = meta["query"].map(normalize_answer)
    first_seen = {}
    for i, q in enumerate(norm_q):
        if q in first_seen:
            uf.union(first_seen[q], i)
        else:
            first_seen[q] = i
    n_emb_links = 0
    if embeddings is not None:
        e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        for start in range(0, n, 1024):
            sims = e[start:start + 1024] @ e.T
            for a, b in zip(*np.nonzero(sims >= near_dup)):
                a = a + start
                if a < b:
                    uf.union(a, b)
                    n_emb_links += 1

    comp = np.array([uf.find(i) for i in range(n)])
    largest = np.bincount(comp).max()
    if largest > max_component_frac * n:
        raise ValueError(f"largest linked component has {largest}/{n} queries (> {max_component_frac:.0%}); "
                         "lower hub_title_max_queries or inspect duplicate questions before splitting")
    rng = np.random.default_rng(seed)
    comps = defaultdict(list)
    for i, c in enumerate(comp):
        comps[c].append(i)
    keys = list(comps)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(comps[k]))  # stable sort: random order within equal sizes

    datasets = sorted(meta["dataset"].unique())
    totals = meta["dataset"].value_counts()
    target = {(s, d): ratios[s] * totals[d] for s in SPLITS for d in datasets}
    fill = defaultdict(float)
    assign = np.empty(n, dtype=object)
    for k in keys:
        members = comps[k]
        counts = Counter(meta.loc[members, "dataset"])
        def worst_ratio(s):
            return max((fill[(s, d)] + c) / max(target[(s, d)], 1e-9) for d, c in counts.items())
        best = min(SPLITS, key=lambda s: (worst_ratio(s), SPLITS.index(s)))
        for d, c in counts.items():
            fill[(best, d)] += c
        assign[members] = best

    out = pd.DataFrame({"qid": meta["qid"], "split": assign, "component": comp})
    report = leakage_report(meta, out, hubs, n_emb_links)
    return out, report


def leakage_report(meta: pd.DataFrame, split_df: pd.DataFrame, hubs: set, n_emb_links: int) -> dict:
    m = meta.merge(split_df, on="qid")
    train_titles = {t for ts in m.loc[m["split"] == "train", "gold_titles"] for t in ts}
    test = m[m["split"] == "test"]
    shares_any = test["gold_titles"].map(lambda ts: any(t in train_titles for t in ts))
    shares_nonhub = test["gold_titles"].map(lambda ts: any(t in train_titles and t not in hubs for t in ts))
    norm = m["query"].map(normalize_answer)
    dup_cross = m.assign(nq=norm).groupby("nq")["split"].nunique().gt(1).sum()
    sizes = split_df["component"].value_counts()
    return {
        "split_sizes": m.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_dict(),
        "n_hub_titles": len(hubs),
        "hub_titles_sample": sorted(hubs)[:50],
        "test_share_any_title_with_train": float(shares_any.mean()) if len(test) else 0.0,
        "test_share_nonhub_title_with_train": float(shares_nonhub.mean()) if len(test) else 0.0,
        "duplicate_questions_across_splits": int(dup_cross),
        "embedding_near_dup_links": int(n_emb_links),
        "largest_component": int(sizes.max()),
        "n_components": int(len(sizes)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--embeddings", default=None, help="optional .npy of question embeddings aligned with qa_meta")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    root = Path(cfg["paths"]["data_root"])
    meta = pd.read_parquet(root / "qa_meta.parquet")
    meta["gold_titles"] = meta["gold_titles"].map(list)
    emb = np.load(args.embeddings) if args.embeddings else None
    s = cfg["split"]
    out, report = grouped_split(meta, {k: s[k] for k in SPLITS}, cfg["seed"], s["hub_title_max_queries"], emb,
                                s["near_dup_cosine"])
    out.to_parquet(root / "splits.parquet", index=False)
    res = Path(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "leakage_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    if report["test_share_nonhub_title_with_train"] > 0:
        raise SystemExit("non-hub title shared between train and test: grouping bug")


if __name__ == "__main__":
    main()
