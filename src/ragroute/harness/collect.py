"""Step: merge all generation caches into one scored long table (input to the oracle table).

Output {results}/runs_scored.parquet, one row per (qid, config_id):
  qid, dataset, task, hops, qtype, split, config_id, retriever, top_k, rerank, generator,
  generated_text, pred, em, f1, acc, contains, quality,
  input_tokens, output_tokens, llm_calls, rerank_passes, index_queries, cost (default weights)

Fails loudly if any (query, config) cell is missing: the oracle needs a complete grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ragroute.config import RagConfig, all_configs, load_cost_weights, load_experiment
from ragroute.cost import scalar_cost, structural_counts
from ragroute.eval.metrics import score_row


def collect(cfg: dict) -> pd.DataFrame:
    data = Path(cfg["paths"]["data_root"])
    cache = Path(cfg["paths"]["cache_dir"]) / "generation"
    meta = pd.read_parquet(data / "qa_meta.parquet")
    splits = pd.read_parquet(data / "splits.parquet")[["qid", "split"]]
    meta = meta.merge(splits, on="qid", validate="1:1")
    frames = []
    for c in all_configs(cfg):
        path = cache / f"{c.config_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing generation cache for {c.config_id}")
        frames.append(pd.read_parquet(path))
    runs = pd.concat(frames, ignore_index=True)
    expected = len(meta) * len(all_configs(cfg))
    if len(runs) != expected:
        raise ValueError(f"grid incomplete: {len(runs)} rows, expected {expected}")
    runs = runs.merge(meta[["qid", "dataset", "task", "hops", "qtype", "split", "answers"]], on="qid", validate="m:1")

    scored = [score_row(t, g, list(a)) for t, g, a in zip(runs["task"], runs["generated_text"], runs["answers"])]
    runs = pd.concat([runs, pd.DataFrame(scored, index=runs.index)], axis=1)

    pool = cfg["grid"]["rerank_pool"]
    struct = pd.DataFrame([{"config_id": c.config_id, "retriever": c.retriever, "top_k": c.top_k,
                            "rerank": c.rerank, **structural_counts(c.config_id, pool)} for c in all_configs(cfg)])
    runs = runs.merge(struct, on="config_id", validate="m:1")
    weights = load_cost_weights()
    runs["cost"] = scalar_cost(runs, weights["weight_sets"][weights["default"]])
    return runs.drop(columns=["answers"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    runs = collect(cfg)
    out = Path(cfg["paths"]["results_dir"]) / "runs_scored.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(out, index=False)
    print(runs.groupby(["dataset", "config_id"])[["em", "f1", "quality", "cost"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
