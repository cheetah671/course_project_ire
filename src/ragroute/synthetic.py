"""SYNTHETIC data for dry-running the analysis pipeline (router, tables, plots) without the cluster.

NEVER report anything produced from this module. Output files go to a directory named `synthetic_*` and
every table built from them is fake by construction. Quality is simulated so that the best config depends
on two latent query properties (multi-hop-ness, lexical match), which lets tests check that a router can
learn per-query structure when it exists.

Usage: python -m ragroute.synthetic --out synthetic_run   (then point DATA_ROOT/RESULTS_ROOT at it)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import all_configs, load_cost_weights, load_experiment
from ragroute.cost import scalar_cost, structural_counts
from ragroute.data.splits import grouped_split
from ragroute.features.query_features import build_query_features

DATASETS = {"nq": ("single", "qa"), "hotpotqa": ("multi", "qa"), "triviaqa": ("single", "qa")}


def make(out: Path, n_per_ds: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cfg = load_experiment()
    data_root, results = out / "data", out / "results"
    (results / "features").mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    cfg["paths"].update(data_root=str(data_root), results_dir=str(results), cache_dir=str(data_root / "cache"),
                        project_dir=str(data_root / "autorag_project"))

    rows = []
    for ds, (hops, task) in DATASETS.items():
        for i in range(n_per_ds):
            mh = rng.beta(5, 2) if hops == "multi" else rng.beta(2, 5)
            lex = rng.uniform()
            words = int(6 + 10 * mh + rng.integers(0, 4))
            q = ("Which " if mh > 0.6 and rng.uniform() < 0.5 else "What ") + " ".join(["word"] * words) + f" item{ds}{i}?"
            qtype = ("comparison" if rng.uniform() < 0.3 else "bridge") if hops == "multi" else "single"
            rows.append({"qid": f"{ds}:{i}", "dataset": ds, "task": task, "hops": hops, "qtype": qtype, "query": q,
                         "answers": ["x"], "gold_titles": [f"T{ds}{i // 3}"], "orig_id": str(i), "lat_mh": mh, "lat_lex": lex})
    meta = pd.DataFrame(rows)
    meta["has_gold"], meta["n_gold"] = True, 1
    splits, _ = grouped_split(meta, {"train": 0.6, "dev": 0.2, "test": 0.2}, seed, hub_max=20)
    meta.drop(columns=["lat_mh", "lat_lex"]).to_parquet(data_root / "qa_meta.parquet", index=False)
    splits.to_parquet(data_root / "splits.parquet", index=False)

    w = load_cost_weights()
    pool = cfg["grid"]["rerank_pool"]
    recs = []
    for r in meta.itertuples():
        for c in all_configs(cfg):
            p = 0.25 + (0.15 if c.generator == "qwen7b" else 0) + (0.05 if c.rerank else 0)
            p += 0.25 * r.lat_mh * (c.top_k == 10) - 0.10 * r.lat_mh * (c.top_k == 3)
            p += {"bm25": 0.25 * r.lat_lex - 0.05, "dense": 0.20 * (1 - r.lat_lex), "hybrid": 0.12}[c.retriever]
            p = float(np.clip(p, 0.02, 0.98))
            em = float(rng.uniform() < p)
            recs.append({"qid": r.qid, "config_id": c.config_id, "generator": c.generator, "retriever": c.retriever,
                         "top_k": c.top_k, "rerank": c.rerank, "dataset": r.dataset, "task": r.task, "hops": r.hops,
                         "qtype": r.qtype, "em": em, "f1": max(em, float(rng.uniform() < 0.2) * 0.5), "acc": np.nan,
                         "input_tokens": 80 + c.top_k * 135 + int(rng.integers(-20, 20)),
                         "output_tokens": int(rng.integers(3, 12)), "generated_text": "x", "pred": "x",
                         **structural_counts(c.config_id, pool)})
    runs = pd.DataFrame(recs).merge(splits[["qid", "split"]], on="qid")
    runs["quality"] = runs["f1"]
    runs["contains"] = runs["em"]
    runs["cost"] = scalar_cost(runs, w["weight_sets"][w["default"]])
    runs.to_parquet(results / "runs_scored.parquet", index=False)

    qf = build_query_features(meta)
    qf.to_parquet(results / "features" / "query_features.parquet", index=False)
    emb = rng.normal(size=(len(meta), 768)).astype(np.float32)
    emb[:, 0] += 3 * meta["lat_mh"].to_numpy()
    np.save(results / "features" / "question_embeddings.npy", emb)
    lex = meta["lat_lex"].to_numpy()
    rf = pd.DataFrame({"qid": meta["qid"], "bm25_s1": 10 + 10 * lex + rng.normal(size=len(meta)),
                       "bm25_gap12_rel": 0.3 * lex + 0.05 * rng.normal(size=len(meta)),
                       "dense_s1": 0.7 + 0.1 * rng.normal(size=len(meta)),
                       "x_jaccard10": 0.2 + 0.3 * lex + 0.05 * rng.normal(size=len(meta))})
    rf.to_parquet(results / "features" / "retrieval_features.parquet", index=False)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_run")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    if not out.name.startswith("synthetic"):
        raise SystemExit("output dir must start with 'synthetic' so fake results are never mistaken for real ones")
    make(out, args.n)
    print(f"synthetic data written to {out} (DO NOT REPORT)")


if __name__ == "__main__":
    main()
