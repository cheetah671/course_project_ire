"""Measure the per-query cost of an index query (weight d) with standard engines, not AutoRAG's wrappers.

Why (Ada jobs 63979 / 64316, 8 Nov 2026): AutoRAG's own retrieval timings were 1,457 ms per query for BM25 (rank_bm25
scores all 500k passages in pure Python) and 452 ms for dense (Chroma lookup plus pandas content fetching). Those
measure the harness, not retrieval: as d they made one BM25 lookup cost more than a 7B answer. Here:
  bm25   Pyserini/Lucene index over the SAME 500k passages (AutoRAG corpus.parquet), default BM25, single thread,
         top-20 per question, N questions x R repeats (run in env ragroute-mining)
  dense  bge-base question encoding on the GPU (batch 64, with the query instruction) + Chroma HNSW top-20 lookup on
         the run's own collection, N questions x R repeats (run in env ragroute-vllm)
Each appends rows {"component": "index", "retriever": ..., "ms_per_query": ...} to the calibration timings file;
scripts/calibrate_costs.py then sets d = mean(bm25, dense) and keeps AutoRAG's value as "d_autorag".

Usage: python -m ragroute.harness.calibrate_index bm25|dense --timings FILE [--n 1000] [--repeats 3] [--workdir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment

POOL = 20


def _questions(cfg, n: int) -> list[str]:
    qa = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "qa.parquet", columns=["query"])
    return qa.sample(n=min(n, len(qa)), random_state=cfg["seed"])["query"].tolist()


def _write(path: str, row: dict) -> None:
    row.update(component="index", host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


def bench_bm25(cfg, timings: str, n: int, repeats: int, workdir: Path) -> None:
    from pyserini.search.lucene import LuceneSearcher

    jsonl, index = workdir / "subcorpus_jsonl", workdir / "subcorpus_lucene"
    if not (index / "_DONE").exists():
        jsonl.mkdir(parents=True, exist_ok=True)
        corpus = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "corpus.parquet", columns=["doc_id", "contents"])
        with open(jsonl / "corpus.jsonl", "w", encoding="utf-8") as f:
            for i, c in zip(corpus["doc_id"], corpus["contents"]):
                f.write(json.dumps({"id": i, "contents": c}, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                        "--input", str(jsonl), "--index", str(index), "--generator", "DefaultLuceneDocumentGenerator",
                        "--threads", "4"], check=True)
        (index / "_DONE").write_text("ok")
    searcher = LuceneSearcher(str(index))
    qs = _questions(cfg, n)
    for q in qs[:20]:  # warm-up (JIT, page cache), discarded
        searcher.search(q, k=POOL)
    for rep in range(repeats):
        t0 = time.perf_counter()
        for q in qs:
            searcher.search(q, k=POOL)
        ms = 1000 * (time.perf_counter() - t0) / len(qs)
        _write(timings, {"retriever": "bm25", "engine": "lucene (pyserini), 1 thread", "rep": rep, "n": len(qs),
                         "n_docs": searcher.num_docs, "ms_per_query": ms})


def bench_dense(cfg, timings: str, n: int, repeats: int) -> None:
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    m = cfg["models"]
    model = SentenceTransformer(m["embedding"], device="cuda")
    coll = chromadb.PersistentClient(path=str(Path(cfg["paths"]["project_dir"]) / "resources" / "chroma")).get_collection("bge_base")
    qs = [m["embedding_query_instruction"] + q for q in _questions(cfg, n)]
    emb = model.encode(qs[:64], batch_size=64, normalize_embeddings=True)  # warm-up, discarded
    coll.query(query_embeddings=emb[:8].tolist(), n_results=POOL)
    for rep in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = model.encode(qs, batch_size=64, normalize_embeddings=True)
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(0, len(qs), 64):
            coll.query(query_embeddings=np.asarray(emb[i:i + 64]).tolist(), n_results=POOL)
        t_ann = time.perf_counter() - t0
        _write(timings, {"retriever": "dense", "engine": "bge-base on GPU (batch 64) + Chroma HNSW", "rep": rep,
                         "n": len(qs), "n_docs": coll.count(), "encode_ms_per_query": 1000 * t_enc / len(qs),
                         "ann_ms_per_query": 1000 * t_ann / len(qs), "ms_per_query": 1000 * (t_enc + t_ann) / len(qs)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["bm25", "dense"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--timings", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workdir", default=None, help="scratch dir for the Lucene index of the sub-corpus (bm25)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    Path(args.timings).parent.mkdir(parents=True, exist_ok=True)
    if args.which == "bm25":
        bench_bm25(cfg, args.timings, args.n, args.repeats, Path(args.workdir or "/tmp/ragroute_index_bench"))
    else:
        bench_dense(cfg, args.timings, args.n, args.repeats)


if __name__ == "__main__":
    main()
