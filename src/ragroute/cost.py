"""Countable cost model.

MARGINAL (per query, per config), counted exactly from the run logs:
  input_tokens, output_tokens  (generator tokenizer; chat template included)
  llm_calls                    (1 per query for every config; judge calls are NOT serving cost)
  rerank_passes                (query-passage pairs scored by the cross-encoder = rerank_pool if reranker on)
  index_queries                (bm25: 1, dense: 1 [includes query encoding], hybrid: 2)

scalar cost = a_in[g]*input_tokens + a_out[g]*output_tokens + b*llm_calls + c*rerank_passes + d*index_queries

OFFLINE (once per corpus): embedding passes, BM25/vector index build time and size. Reported separately and
only folded into a per-query number through an explicit amortisation factor (queries served per build).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ragroute.config import RagConfig

INDEX_QUERIES = {"bm25": 1, "dense": 1, "hybrid": 2}
COMPONENTS = ["input_tokens", "output_tokens", "llm_calls", "rerank_passes", "index_queries"]


def structural_counts(config_id: str, rerank_pool: int) -> dict:
    """Counts fixed by the config alone (no logging needed)."""
    c = RagConfig.from_id(config_id)
    return {
        "llm_calls": 1,
        "rerank_passes": rerank_pool if c.rerank else 0,
        "index_queries": INDEX_QUERIES[c.retriever],
    }


def scalar_cost(df: pd.DataFrame, w: dict) -> np.ndarray:
    """df needs columns: generator, input_tokens, output_tokens, llm_calls, rerank_passes, index_queries."""
    a_in = df["generator"].map(w["a_in"]).to_numpy(float)
    a_out = df["generator"].map(w["a_out"]).to_numpy(float)
    if np.isnan(a_in).any() or np.isnan(a_out).any():
        raise KeyError("cost weights missing a generator key")
    return (
        a_in * df["input_tokens"].to_numpy(float)
        + a_out * df["output_tokens"].to_numpy(float)
        + w["b"] * df["llm_calls"].to_numpy(float)
        + w["c"] * df["rerank_passes"].to_numpy(float)
        + w["d"] * df["index_queries"].to_numpy(float)
    )


def router_overhead(router: str, chosen_retriever: np.ndarray, w: dict, reuse: bool = True) -> np.ndarray:
    """Extra countable cost a router pays on top of the chosen config's own cost.

    Router A (question-only): one question-embedding pass (charged as one index-query-equivalent, d).
        With reuse, it is free if the chosen retriever is dense/hybrid (that embedding is needed anyway).
    Router B (question + post-retrieval probe): BM25 top-pool + dense top-pool = 2 index queries
        (the dense probe includes the question embedding). With reuse, the chosen config's own retrieval
        is served from the probe results (bm25, dense and hybrid=RRF(bm25,dense) are all derivable),
        so its index queries are refunded. reuse=False is the conservative accounting (probe added on top).
    Oracle / static / random: 0.
    """
    chosen_retriever = np.asarray(chosen_retriever)
    own = np.vectorize(INDEX_QUERIES.get)(chosen_retriever).astype(float)
    if router == "A":
        extra = np.ones(len(chosen_retriever))
        if reuse:
            extra[np.isin(chosen_retriever, ["dense", "hybrid"])] = 0.0
        return w["d"] * extra
    if router == "B":
        probe = 2.0 * np.ones(len(chosen_retriever))
        return w["d"] * (probe - own if reuse else probe)
    return np.zeros(len(chosen_retriever))


def amortised_offline_per_query(offline: dict, queries_per_build: float, w_offline_per_gpu_ms: float = 1.0) -> float:
    """offline: {'embed_gpu_ms': ..., 'bm25_build_cpu_ms': ..., ...}. Only GPU-ms terms are summed by default;
    CPU time and index size are reported in their own columns, never silently mixed in."""
    gpu_ms = sum(v for k, v in offline.items() if k.endswith("_gpu_ms"))
    return w_offline_per_gpu_ms * gpu_ms / float(queries_per_build)


def random_weight_draws(w: dict, n: int, factor: float, seed: int) -> list[dict]:
    """Log-uniform multiplicative perturbations of every weight (for the weight-sensitivity analysis)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = lambda: float(np.exp(rng.uniform(-np.log(factor), np.log(factor))))  # noqa: E731
        out.append({
            "a_in": {g: v * m() for g, v in w["a_in"].items()},
            "a_out": {g: v * m() for g, v in w["a_out"].items()},
            "b": w["b"] * m(), "c": w["c"] * m(), "d": w["d"] * m(),
        })
    return out
