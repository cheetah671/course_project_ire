"""Record OFFLINE (once-per-corpus) cost: index sizes on disk + timed, extrapolated build micro-benchmarks.

AutoRAG builds the BM25 pickle and the Chroma index inside start_trial, so their individual build times are
not separable from the trial log. We therefore time each build step on a fixed 50k-passage slice with the
same code/model and extrapolate linearly to the full sub-corpus. Extrapolated numbers are labelled as such.

Writes {results}/offline_cost.json with keys:
  embed_gpu_ms (extrapolated), bm25_build_cpu_ms (extrapolated), chroma_bytes, bm25_bytes, n_passages,
  retrieval_trial_wall_s (from stage_timings.jsonl, includes ingestion + retrieval of all queries)
Usage (on a GPU node): python scripts/record_offline_cost.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from ragroute.config import load_experiment

SLICE = 50_000


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def main():
    cfg = load_experiment()
    project = Path(cfg["paths"]["project_dir"])
    res = Path(cfg["paths"]["results_dir"])
    corpus = pd.read_parquet(project / "data" / "corpus.parquet", columns=["doc_id", "contents"])
    n = len(corpus)
    sl = corpus.head(SLICE)
    out = {"n_passages": n, "slice": SLICE,
           "chroma_bytes": dir_bytes(project / "resources" / "chroma"),
           "bm25_bytes": sum(f.stat().st_size for f in (project / "resources").glob("bm25_*.pkl"))}

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg["models"]["embedding"], device="cuda").half()
    t0 = time.perf_counter()
    m.encode(sl["contents"].tolist(), batch_size=256, normalize_embeddings=True)
    out["embed_gpu_ms"] = (time.perf_counter() - t0) * 1000 * n / SLICE   # 1 GPU
    out["embed_extrapolated"] = True

    from autorag.nodes.lexicalretrieval.bm25 import bm25_ingest
    tmp = res / "tmp_bm25.pkl"
    t0 = time.perf_counter()
    bm25_ingest(str(tmp), sl.assign(metadata=[{}] * len(sl)), bm25_tokenizer=cfg["models"]["bm25_tokenizer"])
    out["bm25_build_cpu_ms"] = (time.perf_counter() - t0) * 1000 * n / SLICE
    out["bm25_extrapolated"] = True
    if tmp.exists():
        os.remove(tmp)

    timings = Path(cfg["paths"]["cache_dir"]) / "stage_timings.jsonl"
    if timings.exists():
        for line in timings.read_text().splitlines():
            r = json.loads(line)
            if r["stage"] == "retrieval_trial_incl_ingest":
                out["retrieval_trial_wall_s"] = r["seconds"]
    (res / "offline_cost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
