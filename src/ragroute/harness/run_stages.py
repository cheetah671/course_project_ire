"""Staged, cached, fully crossed grid runner built on AutoRAG (0.3.27) modules.

Why staged: AutoRAG's Evaluator passes only the best module of each node to the next node, so it cannot
produce outputs for all 24 configs on all queries. We therefore:

  stage `retrieval`  AutoRAG Evaluator.start_trial(retrieval_stage.yaml): ingests BM25 + Chroma once and
                     writes the top-`pool` lists of bm25, dense and hybrid(RRF k=60) for every query.
  stage `rerank`     AutoRAG SentenceTransformerReranker.pure() on each pool once (3 runs, not 12).
  stage `generate`   for one generator: AutoRAG ChatFstring.pure() + Vllm.pure() for all 12 retrieval
                     variants (3 retrievers x 2 top_k x rerank on/off). The model is loaded ONCE.

Cache layout ({cache_dir}):
  retrieval/{bm25,dense,hybrid}.parquet        qid, ids, scores, contents  (top-pool)
  rerank/{bm25,dense,hybrid}.parquet           same, re-sorted by cross-encoder score, + rerank_seconds
  generation/{config_id}.parquet               qid, generated_text, input_tokens, output_tokens, ...
  stage_timings.jsonl                          wall-clock per stage (diagnostic only; latency study is separate)
Each stage skips work whose output file already exists (use --force to redo).

top_k derivation: with reranker off, a config's passages are the first k of its retriever's pool; this
equals AutoRAG retrieving top_k directly because BM25/dense/RRF rankings are deterministic and RRF is
computed over the same 20+20 inputs before truncation. check_equivalence.py verifies this empirically.

Usage:
  python -m ragroute.harness.run_stages retrieval
  python -m ragroute.harness.run_stages rerank
  python -m ragroute.harness.run_stages generate --generator qwen7b
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pandas as pd

from ragroute.config import all_configs, load_experiment

RETRIEVERS = ("bm25", "dense", "hybrid")
NODE_DIR = {"bm25": "lexical_retrieval", "dense": "semantic_retrieval", "hybrid": "hybrid_retrieval"}
SUFFIX = {"bm25": "_lexical", "dense": "_semantic", "hybrid": ""}


def _paths(cfg):
    data = Path(cfg["paths"]["data_root"])
    cache = Path(cfg["paths"]["cache_dir"])
    for sub in ("retrieval", "rerank", "generation"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    return data, cache, Path(cfg["paths"]["project_dir"])


def _log_timing(cache: Path, record: dict):
    record.update(host=socket.gethostname(), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                  slurm_job=os.environ.get("SLURM_JOB_ID"), gpu=gpu_name())
    with open(cache / "stage_timings.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return ";".join(sorted(set(out.stdout.split("\n")) - {""}))
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------ retrieval
def stage_retrieval(cfg, force=False):
    from autorag.evaluator import Evaluator

    data, cache, project = _paths(cfg)
    if not force and all((cache / "retrieval" / f"{r}.parquet").exists() for r in RETRIEVERS):
        print("retrieval cache exists; skipping")
        return
    yaml_path = Path(cfg["paths"]["autorag_yaml_dir"]) / "retrieval_stage.yaml"
    t0 = time.time()
    ev = Evaluator(qa_data_path=str(data / "autorag" / "qa.parquet"),
                   corpus_data_path=str(data / "autorag" / "corpus.parquet"), project_dir=str(project))
    ev.start_trial(str(yaml_path))
    wall = time.time() - t0
    trial = json.loads((project / "trial.json").read_text())[-1]["trial_name"]
    export_retrieval(project, trial, cache)
    _log_timing(cache, {"stage": "retrieval_trial_incl_ingest", "seconds": wall, "trial": trial})


def export_retrieval(project: Path, trial: str, cache: Path):
    qa = pd.read_parquet(project / "data" / "qa.parquet", columns=["qid"])
    node_line = project / trial / "retrieve_node_line"
    for r in RETRIEVERS:
        node = node_line / NODE_DIR[r]
        df = pd.read_parquet(node / "0.parquet")
        s = SUFFIX[r]
        out = pd.DataFrame({
            "qid": qa["qid"].values,
            "ids": df[f"retrieved_ids{s}"].map(list).values,
            "scores": df[f"retrieve_scores{s}"].map(lambda x: [float(v) for v in x]).values,
            "contents": df[f"retrieved_contents{s}"].map(list).values,
        })
        out.to_parquet(cache / "retrieval" / f"{r}.parquet", index=False)
        summ = pd.read_csv(node / "summary.csv")
        _log_timing(cache, {"stage": f"retrieval_{r}", "sec_per_query_autorag": float(summ["execution_time"].iloc[0]),
                            "note": "hybrid time = fusion only"})
        print(f"exported {r}: {len(out)} queries, pool={len(out['ids'].iloc[0])}")


# ------------------------------------------------------------------ rerank
def _qa_frame(project: Path) -> pd.DataFrame:
    return pd.read_parquet(project / "data" / "qa.parquet")


def stage_rerank(cfg, force=False):
    from autorag.nodes.passagereranker import SentenceTransformerReranker

    _, cache, project = _paths(cfg)
    qa = _qa_frame(project)
    pool = cfg["grid"]["rerank_pool"]
    reranker = SentenceTransformerReranker(str(project), model_name=cfg["models"]["reranker"])
    for r in RETRIEVERS:
        out_path = cache / "rerank" / f"{r}.parquet"
        if out_path.exists() and not force:
            continue
        ret = pd.read_parquet(cache / "retrieval" / f"{r}.parquet")
        df = qa.merge(ret, on="qid", how="inner", validate="1:1")
        df = df.rename(columns={"ids": "retrieved_ids", "scores": "retrieve_scores", "contents": "retrieved_contents"})
        t0 = time.time()
        res = reranker.pure(df, top_k=pool, batch=64)
        secs = time.time() - t0
        out = pd.DataFrame({"qid": df["qid"].values, "ids": res["retrieved_ids"].map(list).values,
                            "scores": res["retrieve_scores"].map(lambda x: [float(v) for v in x]).values,
                            "contents": res["retrieved_contents"].map(list).values,
                            "n_pairs": df["retrieved_ids"].map(len).values})
        out.to_parquet(out_path, index=False)
        _log_timing(cache, {"stage": f"rerank_{r}", "seconds": secs, "pairs": int(out["n_pairs"].sum())})
        print(f"reranked {r}: {int(out['n_pairs'].sum())} pairs in {secs:.1f}s")


# ------------------------------------------------------------------ generation
def contexts_for(cache: Path, retriever: str, top_k: int, rerank: bool) -> pd.DataFrame:
    src = "rerank" if rerank else "retrieval"
    df = pd.read_parquet(cache / src / f"{retriever}.parquet")
    return pd.DataFrame({"qid": df["qid"], "retrieved_ids": df["ids"].map(lambda x: list(x)[:top_k]),
                         "retrieved_contents": df["contents"].map(lambda x: list(x)[:top_k]),
                         "retrieve_scores": df["scores"].map(lambda x: list(x)[:top_k])})


def stage_generate(cfg, gen_key: str, force=False, limit: int | None = None, only: str | None = None,
                   subdir: str = "generation"):
    """`only` + `subdir` let the reproducibility check re-run one config into a separate folder
    (e.g. generation_repro/) without touching the main cache."""
    from autorag.nodes.generator import Vllm
    from autorag.nodes.promptmaker import ChatFstring
    from transformers import AutoTokenizer

    data, cache, project = _paths(cfg)
    (cache / subdir).mkdir(parents=True, exist_ok=True)
    qa = _qa_frame(project)
    meta = pd.read_parquet(data / "qa_meta.parquet", columns=["qid", "task"])
    qa = qa.merge(meta, on="qid", how="left", validate="1:1")
    if limit:
        qa = qa.head(limit)
    g = cfg["models"]["generators"][gen_key]
    gen = cfg["generation"]
    configs = [c for c in all_configs(cfg) if c.generator == gen_key and (only is None or c.config_id == only)]
    todo = [c for c in configs if force or not (cache / subdir / f"{c.config_id}.parquet").exists()]
    if not todo:
        print(f"all {len(configs)} configs for {gen_key} cached; skipping")
        return
    tok = AutoTokenizer.from_pretrained(g["hf_id"])
    t_load = time.time()
    llm = Vllm(str(project), llm=g["hf_id"], tensor_parallel_size=g["tensor_parallel_size"], dtype=gen["dtype"],
               max_model_len=gen["max_model_len"], gpu_memory_utilization=gen["gpu_memory_utilization"])
    _log_timing(cache, {"stage": f"load_{gen_key}", "seconds": time.time() - t_load})
    prompter = ChatFstring(str(project))
    for c in todo:
        ctx = contexts_for(cache, c.retriever, c.top_k, c.rerank)
        df = qa.merge(ctx, on="qid", how="inner", validate="1:1")
        prompt_map = {}
        for task, sub in df.groupby("task"):
            sub = sub.reset_index(drop=True)
            p = prompter.pure(sub, prompt=cfg["prompts"][task])
            prompt_map.update(zip(sub["qid"], p["prompts"]))
        df["prompts"] = pd.Series([prompt_map[q] for q in df["qid"]], index=df.index, dtype=object)
        t0 = time.time()
        res = llm.pure(df[["prompts"]].copy(), temperature=gen["temperature"], max_tokens=gen["max_tokens"],
                       seed=gen["seed"])
        secs = time.time() - t0
        in_tok = [len(tok.apply_chat_template(p, add_generation_prompt=True, tokenize=True)) for p in df["prompts"]]
        out = pd.DataFrame({
            "qid": df["qid"].values, "config_id": c.config_id, "generator": gen_key,
            "generated_text": res["generated_texts"].values,
            "output_tokens": res["generated_tokens"].map(len).values,
            "input_tokens": in_tok,
            "retrieved_ids": df["retrieved_ids"].values,
        })
        out.to_parquet(cache / subdir / f"{c.config_id}.parquet", index=False)
        _log_timing(cache, {"stage": f"generate_{c.config_id}", "seconds": secs, "n": len(out),
                            "input_tokens": int(out["input_tokens"].sum()), "output_tokens": int(out["output_tokens"].sum())})
        print(f"{c.config_id}: {len(out)} gens in {secs:.1f}s; in_tok={out['input_tokens'].sum()} out_tok={out['output_tokens'].sum()}")


def check_prompt_token_count(cfg, gen_key: str, n: int = 20):
    """Pilot check: tokenizer-based input count == vLLM's own prompt_token_ids length (run once, week 1)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    _, cache, project = _paths(cfg)
    g = cfg["models"]["generators"][gen_key]
    df = pd.read_parquet(next((cache / "generation").glob(f"*-{gen_key}.parquet")))
    tok = AutoTokenizer.from_pretrained(g["hf_id"])
    qa = _qa_frame(project).set_index("qid")
    ctx = contexts_for(cache, "bm25", 3, False).set_index("qid")
    msgs = []
    for qid in df["qid"].head(n):
        contents = "\n\n".join(ctx.loc[qid, "retrieved_contents"])
        msgs.append([{k: v.format(query=qa.loc[qid, "query"], retrieved_contents=contents) if k == "content" else v
                      for k, v in m.items()} for m in cfg["prompts"]["qa"]])
    llm = LLM(g["hf_id"], tensor_parallel_size=g["tensor_parallel_size"], dtype=cfg["generation"]["dtype"],
              max_model_len=cfg["generation"]["max_model_len"])
    outs = llm.chat(msgs, SamplingParams(max_tokens=1, temperature=0))
    ours = [len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)) for m in msgs]
    theirs = [len(o.prompt_token_ids) for o in outs]
    print("tokenizer vs vLLM prompt lengths equal:", ours == theirs, list(zip(ours, theirs))[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieval", "rerank", "generate", "check_tokens"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N queries only")
    ap.add_argument("--only", default=None, help="run a single config_id")
    ap.add_argument("--subdir", default="generation", help="cache sub-folder for outputs")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    if args.stage == "retrieval":
        stage_retrieval(cfg, args.force)
    elif args.stage == "rerank":
        stage_rerank(cfg, args.force)
    elif args.stage == "generate":
        if args.limit and args.subdir == "generation":
            raise SystemExit("--limit writes partial outputs; use --subdir <other> so the main cache stays complete")
        stage_generate(cfg, args.generator, args.force, args.limit, args.only, args.subdir)
    else:
        check_prompt_token_count(cfg, args.generator)


if __name__ == "__main__":
    main()
