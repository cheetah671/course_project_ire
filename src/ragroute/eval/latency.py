"""Latency study (SECONDARY metric). Rules enforced here:
  * one SLURM job = one session on one node; every row carries session_id, host and GPU name;
    tables.latency_summary refuses to compare rows from different sessions
  * batch size 1 (single-request serving latency), warm-up queries discarded
  * R repeats; config order is re-shuffled (seeded) in every repeat to spread drift/thermal effects
  * both generators run in the same job: this script is called once per generator inside one sbatch script

Uses AutoRAG modules directly (BM25, VectorDB, HybridRRF, SentenceTransformerReranker, ChatFstring, Vllm).
Caveat: AutoRAG BM25 is rank_bm25 in pure Python (seconds per query on 1M passages); measured BM25 latency
reflects that implementation, not BM25 in a production engine such as Lucene. Say so in the report.

Usage (inside slurm/06_latency.sbatch):
  python -m ragroute.eval.latency --generator qwen1p5b --session $SLURM_JOB_ID --n 200 --repeats 3
"""
from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import all_configs, load_experiment
from ragroute.harness.make_autorag_yaml import VDB_NAME
from ragroute.harness.run_stages import gpu_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    from autorag.nodes.generator import Vllm
    from autorag.nodes.hybridretrieval import HybridRRF
    from autorag.nodes.lexicalretrieval import BM25
    from autorag.nodes.passagereranker import SentenceTransformerReranker
    from autorag.nodes.promptmaker import ChatFstring
    from autorag.nodes.semanticretrieval import VectorDB

    cfg = load_experiment(args.config)
    project = cfg["paths"]["project_dir"]
    res = Path(cfg["paths"]["results_dir"])
    g = cfg["models"]["generators"][args.generator]
    gen = cfg["generation"]
    pool, rrf_k = cfg["grid"]["rerank_pool"], cfg["grid"]["rrf_k"]

    qa = pd.read_parquet(Path(project) / "data" / "qa.parquet")
    meta = pd.read_parquet(Path(cfg["paths"]["data_root"]) / "qa_meta.parquet", columns=["qid", "task"])
    splits = pd.read_parquet(Path(cfg["paths"]["data_root"]) / "splits.parquet")
    qa = qa.merge(meta, on="qid").merge(splits[["qid", "split"]], on="qid")
    test = qa[(qa["split"] == "test") & (qa["task"] == "qa")].sample(n=args.n + args.warmup, random_state=cfg["seed"])

    bm25 = BM25(project, bm25_tokenizer=cfg["models"]["bm25_tokenizer"])
    vdb = VectorDB(project, vectordb=VDB_NAME)
    hybrid = HybridRRF(project)
    reranker = SentenceTransformerReranker(project, model_name=cfg["models"]["reranker"])
    prompter = ChatFstring(project)
    llm = Vllm(project, llm=g["hf_id"], tensor_parallel_size=g["tensor_parallel_size"], dtype=gen["dtype"],
               max_model_len=gen["max_model_len"], gpu_memory_utilization=gen["gpu_memory_utilization"])
    configs = [c for c in all_configs(cfg) if c.generator == args.generator]

    def one(row: pd.DataFrame, c) -> dict:
        t = {}
        depth = pool if c.rerank else c.top_k
        t0 = time.perf_counter()
        if c.retriever == "bm25":
            r = bm25.pure(row.copy(), top_k=depth)
            r = r.rename(columns=lambda x: x.replace("_lexical", ""))
        elif c.retriever == "dense":
            r = vdb.pure(row.copy(), top_k=depth)
            r = r.rename(columns=lambda x: x.replace("_semantic", ""))
        else:
            lex = bm25.pure(row.copy(), top_k=pool)
            sem = vdb.pure(row.copy(), top_k=pool)
            both = pd.concat([row.reset_index(drop=True), lex, sem], axis=1)
            r = hybrid.pure(both, top_k=depth, weight=rrf_k)
        t["t_retrieval"] = time.perf_counter() - t0
        df = pd.concat([row.reset_index(drop=True), r.reset_index(drop=True)], axis=1)
        t0 = time.perf_counter()
        if c.rerank:
            rr = reranker.pure(df, top_k=c.top_k)
            df = pd.concat([row.reset_index(drop=True), rr], axis=1)
        t["t_rerank"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = prompter.pure(df, prompt=cfg["prompts"]["qa"])
        t["t_prompt"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        llm.pure(p, temperature=gen["temperature"], max_tokens=gen["max_tokens"])
        t["t_generate"] = time.perf_counter() - t0
        t["t_total"] = sum(t.values())
        return t

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    cols = ["qid", "query", "retrieval_gt", "generation_gt"]
    for rep in range(args.repeats):
        for c in rng.permutation(np.array(configs, dtype=object)):
            for i, (_, q) in enumerate(test.iterrows()):
                t = one(pd.DataFrame([q[cols]]), c)
                if i < args.warmup:
                    continue
                rows.append({"session_id": args.session, "host": socket.gethostname(), "gpu": gpu_name(),
                             "generator": args.generator, "config_id": c.config_id, "repeat": rep, "qid": q["qid"], **t})
            print(f"repeat {rep} {c.config_id} done")
    out = res / "latency.parquet"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old[old["session_id"] == args.session], df])  # keep only this session
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
