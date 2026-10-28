"""Oracle table and oracle policies.

Oracle table = one row per (query, config) with quality in [0,1] and scalar cost. Wide views:
  Q[q, c] = quality, C[q, c] = cost   (queries x 24 configs)

Oracle policies (all computed with TRUE labels, so they are upper bounds):
  oracle_best        argmax_c Q[q,c], ties -> cheapest            (max-quality ceiling)
  oracle_cheapest_ok cheapest c with Q[q,c] >= tau (EM-style success), else oracle_best
  oracle_lambda(l)   argmax_c Q[q,c] - l * C[q,c] / cost_scale    (oracle Pareto curve, swept over l)

cost_scale is ONE constant (median cost of the most expensive static config on train) so lambda is unitless.
It is a fixed rescaling, not a per-query normalisation, so it cannot reorder configs within a query.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wide(runs: pd.DataFrame, value: str) -> pd.DataFrame:
    return runs.pivot(index="qid", columns="config_id", values=value).sort_index(axis=0).sort_index(axis=1)


def cost_scale(runs: pd.DataFrame, split: str = "train") -> float:
    sub = runs[runs["split"] == split]
    return float(sub.groupby("config_id")["cost"].median().max())


def choose(score: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Row-wise argmax of score, ties broken by lower cost, then by column order (deterministic)."""
    n, _ = score.shape
    out = np.empty(n, dtype=int)
    for i in range(n):
        best = np.flatnonzero(np.isclose(score[i], score[i].max(), atol=1e-12))
        out[i] = best[np.argmin(cost[i, best])]
    return out


def oracle_best(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    return choose(Q, C)


def oracle_cheapest_ok(Q: np.ndarray, C: np.ndarray, tau: float = 1.0) -> np.ndarray:
    ok = Q >= tau - 1e-12
    masked_cost = np.where(ok, C, np.inf)
    pick = masked_cost.argmin(axis=1)
    none_ok = ~ok.any(axis=1)
    pick[none_ok] = choose(Q[none_ok], C[none_ok]) if none_ok.any() else pick[none_ok]
    return pick


def oracle_lambda(Q: np.ndarray, C: np.ndarray, lam: float, scale: float) -> np.ndarray:
    return choose(Q - lam * C / scale, C)


def realised(Q: np.ndarray, C: np.ndarray, pick: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(pick))
    return Q[rows, pick], C[rows, pick]


def oracle_summary(runs: pd.DataFrame, lambdas, split: str = "test") -> pd.DataFrame:
    """Mean quality/cost of oracle policies on one split, plus label statistics used in the report."""
    sub = runs[runs["split"] == split]
    Qd, Cd = wide(sub, "quality"), wide(sub, "cost")
    Q, C = Qd.to_numpy(), Cd.to_numpy()
    scale = cost_scale(runs)
    rows = []
    for name, pick in [("oracle_best", oracle_best(Q, C)), ("oracle_cheapest_ok", oracle_cheapest_ok(Q, C))]:
        q, c = realised(Q, C, pick)
        rows.append({"policy": name, "lambda": np.nan, "quality": q.mean(), "cost": c.mean()})
    for lam in lambdas:
        q, c = realised(Q, C, oracle_lambda(Q, C, lam, scale))
        rows.append({"policy": "oracle_lambda", "lambda": lam, "quality": q.mean(), "cost": c.mean()})
    return pd.DataFrame(rows)


def label_statistics(runs: pd.DataFrame, success_tau: float = 1.0) -> pd.DataFrame:
    """Per dataset: share of queries where all configs fail / all succeed / some succeed (router-relevant),
    and how often each config is the oracle_best pick (label imbalance diagnostics)."""
    out = []
    for ds, sub in runs.groupby("dataset"):
        Q = wide(sub, "quality").to_numpy()
        C = wide(sub, "cost").to_numpy()
        ok = Q >= success_tau - 1e-12
        out.append({
            "dataset": ds, "n": len(Q),
            "all_fail": float((~ok).all(axis=1).mean()),
            "all_succeed": float(ok.all(axis=1).mean()),
            "mixed": float((ok.any(axis=1) & ~ok.all(axis=1)).mean()),
            "n_distinct_best": int(len(np.unique(oracle_best(Q, C)))),
            "top_label_share": float(np.bincount(oracle_best(Q, C)).max() / len(Q)),
        })
    return pd.DataFrame(out)
