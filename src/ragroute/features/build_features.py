"""Build router feature files.

Outputs in {results}/features/:
  query_features.parquet       group 1 (surface/cue) + group 2 (data)
  question_embeddings.npy      aligned with query_features.qid (group 1, embedding part)
  retrieval_features.parquet   Router B probe signals

Usage: python -m ragroute.features.build_features [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment
from ragroute.features.query_features import build_query_features, embed_questions
from ragroute.features.retrieval_features import build_retrieval_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    data = Path(cfg["paths"]["data_root"])
    cache = Path(cfg["paths"]["cache_dir"])
    out = Path(cfg["paths"]["results_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(data / "qa_meta.parquet").sort_values("qid").reset_index(drop=True)
    qf = build_query_features(meta)
    qf.to_parquet(out / "query_features.parquet", index=False)
    emb = embed_questions(meta["query"].tolist(), cfg["models"]["embedding"],
                          cfg["models"]["embedding_query_instruction"], args.device)
    np.save(out / "question_embeddings.npy", emb.astype(np.float32))

    bm25 = pd.read_parquet(cache / "retrieval" / "bm25.parquet")
    dense = pd.read_parquet(cache / "retrieval" / "dense.parquet")
    rf = build_retrieval_features(meta[["qid", "query"]], bm25, dense)
    rf.to_parquet(out / "retrieval_features.parquet", index=False)
    print(f"query features {qf.shape}, embeddings {emb.shape}, retrieval features {rf.shape}")


if __name__ == "__main__":
    main()
