"""Latency study (SECONDARY metric): end-to-end seconds per question, p50/p95, for all 24 configurations.

Rules (TA feedback; docs/02 D18):
  * one SLURM job = one session on one node; every row carries session_id, host and GPU; the summary refuses rows
    from more than one session
  * batch size 1 (one question at a time: serving latency, not throughput); warm-up passes discarded
  * R repeats; the configuration order is re-shuffled (seeded) in every repeat to spread drift and thermal effects
  * both generators run in the same job, one after the other

What is timed, per (question, configuration, repeat):
  retrieval   bm25    Lucene (Pyserini) BM25 over the run's own 500k-passage sub-corpus, single thread, top-depth.
                      Lucene needs Java and lives in env ragroute-mining, so BM25 is timed in a first step of the
                      SAME job (`bm25` mode) for every (question, repeat, depth) and looked up afterwards.
              dense   bge-base question encoding on the GPU + Chroma HNSW top-depth lookup on the run's collection
              hybrid  bm25 (depth 20) + dense (depth 20) + reciprocal-rank fusion (k = 60) of the two pools
              Why not AutoRAG's retrieval modules: its BM25 is pure-Python rank_bm25 (~1 s per question over 500k
              passages), which measures the harness, not BM25 (same reason as the index-cost weight, report §3.5).
  rerank      cross-encoder over the 20-passage pool of that retriever (reranker-on configurations only)
  generate    one vLLM chat request with the exact prompt of the main run (task prompt + the configuration's
              passages from the stage cache), greedy, at most 32 output tokens
  total       retrieval + rerank + generate (prompt formatting is negligible and not counted)

Modes (slurm/ada/jobL_latency_gpu.sbatch runs all three in one job):
  python -m ragroute.eval.latency bm25      --session S [--workdir DIR]        (env ragroute-mining)
  python -m ragroute.eval.latency pipeline  --session S --generator qwen1p5b   (env ragroute-vllm, GPU)
  python -m ragroute.eval.latency summarise --session S
Outputs: {results}/latency_parts/{S}/... (one file per generator x repeat x configuration, written as it finishes),
{results}/latency.parquet (all rows of session S) and {results}/tables/T10_latency.csv (p50/p95 per configuration).
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import all_configs, load_experiment

DEPTHS = (3, 10, 20)


# ------------------------------------------------------------------ pure helpers (tested locally)
def sample_questions(meta: pd.DataFrame, splits: pd.DataFrame, n_per_dataset: int, seed: int) -> pd.DataFrame:
    """Test-split questions, the same number from every dataset, deterministic."""
    m = meta.merge(splits[["qid", "split"]], on="qid", validate="1:1")
    m = m[m["split"] == "test"]
    parts = [g.sample(n=min(n_per_dataset, len(g)), random_state=seed) for _, g in m.groupby("dataset", sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def rrf(a_ids, b_ids, k: int, depth: int) -> list:
    """Reciprocal-rank fusion of two ranked id lists (ranks start at 1), top `depth` ids."""
    score: dict = {}
    for lst in (a_ids, b_ids):
        for r, d in enumerate(lst):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda d: (-score[d], str(d)))[:depth]


def retrieval_depth(c, pool: int) -> int:
    return pool if c.rerank else c.top_k


def summarise_rows(lat: pd.DataFrame) -> pd.DataFrame:
    """T10: p50/p95 of end-to-end seconds per configuration, plus the median of each component."""
    if lat["session_id"].nunique() != 1:
        raise ValueError("latency rows from >1 session: never compare timings across sessions")
    g = lat.groupby(["generator", "config_id"])
    out = pd.DataFrame({
        "p50_s": g["t_total"].quantile(0.5), "p95_s": g["t_total"].quantile(0.95),
        "retrieval_p50_s": g["t_retrieval"].quantile(0.5), "rerank_p50_s": g["t_rerank"].quantile(0.5),
        "generate_p50_s": g["t_generate"].quantile(0.5), "generate_p95_s": g["t_generate"].quantile(0.95),
        "output_tokens_mean": g["output_tokens"].mean(), "n": g.size(),
    }).reset_index()
    out.insert(0, "session_id", lat["session_id"].iloc[0])
    out["host"] = lat["host"].iloc[0]
    out["gpu"] = lat["gpu"].iloc[0]
    return out.sort_values(["generator", "p50_s"]).reset_index(drop=True)


# ------------------------------------------------------------------ shared I/O
def _setup(cfg, n_per_dataset: int):
    data = Path(cfg["paths"]["data_root"])
    res = Path(cfg["paths"]["results_dir"])
    meta = pd.read_parquet(data / "qa_meta.parquet", columns=["qid", "dataset", "task"])
    splits = pd.read_parquet(data / "splits.parquet")
    qa = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "qa.parquet", columns=["qid", "query"])
    qs = sample_questions(meta, splits, n_per_dataset, cfg["seed"]).merge(qa, on="qid", validate="1:1")
    return res, qs


def _parts_dir(res: Path, session: str) -> Path:
    d = res / "latency_parts" / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp() -> dict:
    from ragroute.harness.run_stages import gpu_name

    return {"host": socket.gethostname(), "gpu": gpu_name(), "slurm_job": os.environ.get("SLURM_JOB_ID")}


# ------------------------------------------------------------------ mode: bm25 (env ragroute-mining)
def run_bm25(cfg, session: str, n_per_dataset: int, repeats: int, workdir: Path, warmup: int = 20):
    from pyserini.search.lucene import LuceneSearcher

    from ragroute.harness.calibrate_index import subcorpus_lucene_index

    res, qs = _setup(cfg, n_per_dataset)
    searcher = LuceneSearcher(str(subcorpus_lucene_index(cfg, workdir, threads=8)))
    for q in qs["query"].head(warmup):
        searcher.search(q, k=max(DEPTHS))
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for rep in range(repeats):
        for i in rng.permutation(len(qs)):
            qid, query = qs.at[i, "qid"], qs.at[i, "query"]
            for depth in rng.permutation(DEPTHS):
                t0 = time.perf_counter()
                searcher.search(query, k=int(depth))
                rows.append({"qid": qid, "repeat": rep, "depth": int(depth), "t_bm25": time.perf_counter() - t0})
    out = pd.DataFrame(rows).assign(session_id=session, n_docs=searcher.num_docs, **_stamp())
    out.to_parquet(_parts_dir(res, session) / "bm25.parquet", index=False)
    print(f"bm25: {len(out)} searches, median {1000 * out['t_bm25'].median():.2f} ms")


# ------------------------------------------------------------------ mode: pipeline (env ragroute-vllm, GPU)
def run_pipeline(cfg, session: str, gen_key: str, n_per_dataset: int, repeats: int, warmup: int = 5):
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from vllm import LLM, SamplingParams

    from ragroute.harness.run_stages import contexts_for, gen_dtype

    res, qs = _setup(cfg, n_per_dataset)
    parts = _parts_dir(res, session)
    bm25_path = parts / "bm25.parquet"
    if not bm25_path.exists():
        raise SystemExit(f"{bm25_path} missing: run the bm25 mode first, in the same job")
    bm = pd.read_parquet(bm25_path)
    if (bm["session_id"] != session).any():
        raise SystemExit("bm25 timings belong to another session")
    t_bm25 = {(q, r, d): t for q, r, d, t in bm[["qid", "repeat", "depth", "t_bm25"]].itertuples(index=False)}

    m, gen, grid = cfg["models"], cfg["generation"], cfg["grid"]
    pool, rrf_k = grid["rerank_pool"], grid["rrf_k"]
    cache, project = Path(cfg["paths"]["cache_dir"]), Path(cfg["paths"]["project_dir"])
    wanted = set(qs["qid"])
    pools = {r: pd.read_parquet(cache / "retrieval" / f"{r}.parquet").set_index("qid").loc[list(wanted)]
             for r in ("bm25", "dense", "hybrid")}
    configs = [c for c in all_configs(cfg) if c.generator == gen_key]
    prompts = {}  # (config_id, qid) -> chat messages, exactly as in the main run (run_stages._generate_configs)
    for c in configs:
        ctx = contexts_for(cache, c.retriever, c.top_k, c.rerank).set_index("qid")
        for qid, query, task in qs[["qid", "query", "task"]].itertuples(index=False):
            passages = "\n\n".join(ctx.at[qid, "retrieved_contents"])
            prompts[(c.config_id, qid)] = [
                {k: (v.format(query=query, retrieved_contents=passages) if k == "content" else v) for k, v in msg.items()}
                for msg in cfg["prompts"][task]]

    # Small models first; vLLM's engine then gets a fixed share of what is left on the 11 GB card. 0.65 leaves room
    # for bge + the cross-encoder in this process (job D needed 0.6 next to AutoRAG's embedder and reranker).
    embedder = SentenceTransformer(m["embedding"], device="cuda")
    reranker = CrossEncoder(m["reranker"], device="cuda")
    coll = chromadb.PersistentClient(path=str(project / "resources" / "chroma")).get_collection("bge_base")
    g = m["generators"][gen_key]
    llm = LLM(g.get("vllm_model", g["hf_id"]), tensor_parallel_size=g["tensor_parallel_size"], dtype=gen_dtype(cfg, gen_key),
              max_model_len=gen["max_model_len"], seed=gen["seed"],
              gpu_memory_utilization=float(os.environ.get("RAGROUTE_LAT_GPU_UTIL", 0.65)),
              max_num_seqs=min(8, gen["max_num_seqs"]),  # batch size 1 here; fewer warm-up sequences, less memory
              max_num_batched_tokens=gen["max_num_batched_tokens"])
    sp = SamplingParams(temperature=gen["temperature"], max_tokens=gen["max_tokens"], seed=gen["seed"])
    instr = m["embedding_query_instruction"]

    def dense(query: str, depth: int) -> float:
        t0 = time.perf_counter()
        e = embedder.encode([instr + query], normalize_embeddings=True)  # returns numpy: GPU work is finished
        coll.query(query_embeddings=np.asarray(e).tolist(), n_results=depth)
        return time.perf_counter() - t0

    def one(c, qid: str, query: str, rep: int) -> dict:
        depth = retrieval_depth(c, pool)
        if c.retriever == "bm25":
            t_ret = t_bm25[(qid, rep, depth)]
        elif c.retriever == "dense":
            t_ret = dense(query, depth)
        else:
            t_ret = t_bm25[(qid, rep, pool)] + dense(query, pool)
            t0 = time.perf_counter()
            rrf(pools["bm25"].at[qid, "ids"], pools["dense"].at[qid, "ids"], rrf_k, depth)
            t_ret += time.perf_counter() - t0
        t_rr = 0.0
        if c.rerank:
            pairs = [(query, p) for p in list(pools[c.retriever].at[qid, "contents"])[:pool]]
            t0 = time.perf_counter()
            reranker.predict(pairs, batch_size=pool, show_progress_bar=False)
            t_rr = time.perf_counter() - t0
        t0 = time.perf_counter()
        o = llm.chat([prompts[(c.config_id, qid)]], sp, use_tqdm=False)
        t_gen = time.perf_counter() - t0
        return {"t_retrieval": t_ret, "t_rerank": t_rr, "t_generate": t_gen, "t_total": t_ret + t_rr + t_gen,
                "output_tokens": len(o[0].outputs[0].token_ids)}

    for c in configs:  # warm-up: CUDA kernels, Chroma pages, vLLM graphs; discarded
        for qid, query in qs[["qid", "query"]].head(warmup).itertuples(index=False):
            one(c, qid, query, 0)
    stamp = _stamp()
    rng = np.random.default_rng([cfg["seed"], int.from_bytes(gen_key.encode(), "little") % 2**32])
    for rep in range(repeats):
        for pos, c in enumerate(rng.permutation(np.array(configs, dtype=object))):
            f = parts / f"{gen_key}_r{rep}_{c.config_id}.parquet"
            if f.exists():
                continue
            rows = [{"qid": qid, "dataset": ds, **one(c, qid, query, rep)}
                    for qid, query, ds in qs[["qid", "query", "dataset"]].itertuples(index=False)]
            (pd.DataFrame(rows).assign(session_id=session, generator=gen_key, config_id=c.config_id, repeat=rep,
                                       order_in_repeat=pos, **stamp)
             .to_parquet(f, index=False))
            print(f"repeat {rep} {c.config_id}: median {pd.DataFrame(rows)['t_total'].median():.3f} s", flush=True)


# ------------------------------------------------------------------ mode: summarise
def run_summarise(cfg, session: str):
    res = Path(cfg["paths"]["results_dir"])
    files = sorted(f for f in _parts_dir(res, session).glob("*.parquet") if f.name != "bm25.parquet")
    if not files:
        raise SystemExit("no latency parts for this session")
    lat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lat.to_parquet(res / "latency.parquet", index=False)
    t = summarise_rows(lat)
    (res / "tables").mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "tables" / "T10_latency.csv", index=False)
    n_cfg = lat["config_id"].nunique()
    print(f"{len(lat)} rows, {n_cfg} configurations, session {session}")
    print(t[["config_id", "p50_s", "p95_s", "retrieval_p50_s", "rerank_p50_s", "generate_p50_s"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bm25", "pipeline", "summarise"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--session", required=True)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--n-per-dataset", type=int, default=40)   # 5 datasets x 40 = 200 test questions
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workdir", default=None, help="node-local dir for the sub-corpus Lucene index (bm25 mode)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    if args.mode == "bm25":
        run_bm25(cfg, args.session, args.n_per_dataset, args.repeats, Path(args.workdir or "/tmp/ragroute_index_bench"))
    elif args.mode == "pipeline":
        if not args.generator:
            ap.error("--generator is required for pipeline mode")
        run_pipeline(cfg, args.session, args.generator, args.n_per_dataset, args.repeats)
    else:
        run_summarise(cfg, args.session)


if __name__ == "__main__":
    main()
