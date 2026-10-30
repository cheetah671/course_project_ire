"""Measure the timings behind the cost weights (configs/cost_weights.yaml), BEFORE the full run.

The countable cost components (tokens, LLM calls, reranker pairs, index queries) are exact. Only the weights
that turn them into one scalar cost are estimates until measured here. Run this once on the SAME GPU type and
serving engine as the full run (RTX 2080 Ti, vLLM), on the pilot's cached passages, then fit the weights with
`scripts/calibrate_costs.py`.

Components
  generate  per generator: the same N real prompts at k in {1, 3, 10} passages, each with a forced output length
            M in {1, 32} tokens (min_tokens = max_tokens = M, EOS ignored), R repeats, cell order shuffled per
            repeat, after a discarded warm-up. Changing k moves only input tokens and changing M moves only
            output tokens, so the fit can separate a_in (per prompt token), a_out (per generated token) and
            b (per call). Prefix caching is OFF: the k=3 and k=10 prompts share a long prefix, and a cache hit
            would make prompt tokens look free. The cost model counts every prompt token.
  rerank    AutoRAG's SentenceTransformerReranker (the same code path as the run) on N queries x pool pairs,
            R repeats -> c (per pair).
  index     per-query retrieval time is not re-measured: AutoRAG records it in the retrieval stage
            (stage_timings.jsonl), which `calibrate_costs.py` reads.

Output: {results}/calibration/calibration_timings.jsonl, one row per timed batch (with host, GPU, job id).
Usage:
  python -m ragroute.harness.calibrate generate --generator qwen1p5b [--n 200] [--repeats 3]
  python -m ragroute.harness.calibrate rerank [--n 200] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment
from ragroute.harness.run_stages import _paths, _qa_frame, gpu_name

K_LEVELS = (1, 3, 10)
M_LEVELS = (1, 32)


def _out_file(cfg) -> Path:
    out = Path(cfg["paths"]["results_dir"]) / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    return out / "calibration_timings.jsonl"


def _write(cfg, row: dict):
    row.update(host=socket.gethostname(), gpu=gpu_name(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(_out_file(cfg), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(row)


def _sample(cfg, n: int) -> pd.DataFrame:
    """N queries (seeded) with their task and BM25 top-pool passages from the run's retrieval cache."""
    data, cache, project = _paths(cfg)
    qa = _qa_frame(project).merge(pd.read_parquet(data / "qa_meta.parquet", columns=["qid", "task"]), on="qid")
    ret = pd.read_parquet(cache / "retrieval" / "bm25.parquet")
    df = qa.merge(ret, on="qid", how="inner", validate="1:1")
    return df.sample(n=min(n, len(df)), random_state=cfg["seed"]).reset_index(drop=True)


def build_messages(cfg, df: pd.DataFrame, k: int) -> list:
    """Chat messages exactly as the run's prompt templates build them, with the first k passages."""
    msgs = []
    for query, task, contents in zip(df["query"], df["task"], df["contents"]):
        ctx = "\n\n".join(list(contents)[:k])
        msgs.append([{**m, "content": m["content"].format(query=query, retrieved_contents=ctx)}
                     for m in cfg["prompts"][task]])
    return msgs


def _cells(repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    for rep in range(repeats):
        cells = [(k, m) for k in K_LEVELS for m in M_LEVELS]
        rng.shuffle(cells)
        for k, m in cells:
            yield rep, k, m


def calibrate_generate(cfg, gen_key: str, n: int, repeats: int):
    g = cfg["models"]["generators"][gen_key]
    gen = cfg["generation"]
    df = _sample(cfg, n)
    msgs = {k: build_messages(cfg, df, k) for k in K_LEVELS}
    base = {"component": "generate", "generator": gen_key, "backend": gen["backend"], "n_calls": len(df),
            "gpus": int(g.get("tensor_parallel_size", 1))}

    if gen["backend"] == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(g.get("vllm_model", g["hf_id"]), tensor_parallel_size=g["tensor_parallel_size"], dtype=gen["dtype"],
                  max_model_len=gen["max_model_len"], gpu_memory_utilization=gen["gpu_memory_utilization"],
                  max_num_seqs=gen["max_num_seqs"], max_num_batched_tokens=gen["max_num_batched_tokens"],
                  enable_prefix_caching=False, seed=gen["seed"])
        llm.chat(msgs[3][:16], SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)  # warm-up, discarded

        def run_batch(k, m):
            sp = SamplingParams(temperature=0, max_tokens=m, min_tokens=m, ignore_eos=True, seed=gen["seed"])
            t0 = time.perf_counter()
            outs = llm.chat(msgs[k], sp, use_tqdm=False)
            secs = time.perf_counter() - t0
            return secs, sum(len(o.prompt_token_ids) for o in outs), sum(len(o.outputs[0].token_ids) for o in outs)

        for rep, k, m in _cells(repeats, cfg["seed"]):
            secs, tin, tout = run_batch(k, m)
            _write(cfg, {**base, "rep": rep, "k": k, "max_tokens": m, "seconds": secs,
                         "input_tokens": tin, "output_tokens": tout})
    elif gen["backend"] == "llamacpp":
        import requests

        from ragroute.harness.llamacpp import LlamaServer, ensure_gguf

        lc = gen["llamacpp"]
        _, cache, _ = _paths(cfg)
        with LlamaServer(lc["server_bin"], ensure_gguf(g["gguf_repo"], g["gguf_pattern"]), lc["port"], lc["parallel"],
                         lc["ctx_per_slot"], lc["n_gpu_layers"], gen["seed"],
                         str(cache / f"llama_server_{gen_key}_calibrate.log"), lib_path=lc.get("lib_path")) as srv:
            def one(messages, m):
                # cache_prompt=False: no prompt-prefix reuse between requests (see module docstring)
                r = requests.post(f"{srv.url}/v1/chat/completions", timeout=600,
                                  json={"messages": messages, "max_tokens": m, "temperature": 0, "seed": gen["seed"],
                                        "ignore_eos": True, "cache_prompt": False}).json()["usage"]
                return r["prompt_tokens"], r["completion_tokens"]

            with ThreadPoolExecutor(lc["parallel"]) as ex:  # same concurrency as the run
                list(ex.map(lambda x: one(x, 4), msgs[3][:16]))  # warm-up, discarded
                for rep, k, m in _cells(repeats, cfg["seed"]):
                    t0 = time.perf_counter()
                    usage = list(ex.map(lambda x, m=m: one(x, m), msgs[k]))
                    secs = time.perf_counter() - t0
                    _write(cfg, {**base, "rep": rep, "k": k, "max_tokens": m, "seconds": secs,
                                 "input_tokens": int(sum(u[0] for u in usage)),
                                 "output_tokens": int(sum(u[1] for u in usage))})
    else:
        raise ValueError(f"unknown generation backend {gen['backend']}")


def calibrate_rerank(cfg, n: int, repeats: int):
    import torch
    from autorag.nodes.passagereranker import SentenceTransformerReranker

    _, _, project = _paths(cfg)
    df = _sample(cfg, n).rename(columns={"ids": "retrieved_ids", "scores": "retrieve_scores",
                                         "contents": "retrieved_contents"})
    pool = cfg["grid"]["rerank_pool"]
    reranker = SentenceTransformerReranker(str(project), model_name=cfg["models"]["reranker"])
    reranker.pure(df.head(8).copy(), top_k=pool, batch=64)  # warm-up, discarded
    pairs = int(df["retrieved_ids"].map(len).sum())
    for rep in range(repeats):
        t0 = time.perf_counter()
        reranker.pure(df.copy(), top_k=pool, batch=64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _write(cfg, {"component": "rerank", "rep": rep, "pairs": pairs, "n_queries": len(df),
                     "seconds": time.perf_counter() - t0, "gpus": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("component", choices=["generate", "rerank"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--n", type=int, default=200, help="queries per timed batch")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    if args.component == "generate":
        if not args.generator:
            raise SystemExit("--generator is required for generate")
        calibrate_generate(cfg, args.generator, args.n, args.repeats)
    else:
        calibrate_rerank(cfg, args.n, args.repeats)


if __name__ == "__main__":
    main()
