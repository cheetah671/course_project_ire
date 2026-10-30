"""Routing policies and baselines over a (queries x configs) quality/cost grid. Pure numpy, unit-tested.

Every policy returns per-query config indices (`pick`). Realised quality and cost are always looked up
in the TRUE grid; routers only influence which cell is read. Router overhead (Router A embedding,
Router B probe) is added by `realise`.

Baselines are always FIT ON TRAIN and applied to dev/test (selecting the "best static" on test would be
an optimistic oracle; we report it separately as `static_best_on_eval` for reference only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragroute.oracle import choose


@dataclass
class Grid:
    qids: np.ndarray
    dataset: np.ndarray
    hops: np.ndarray
    qtype: np.ndarray
    Q: np.ndarray        # (n, C) true quality
    C: np.ndarray        # (n, C) true cost
    config_ids: list
    task: np.ndarray | None = None   # qa | verification | slot_filling (None: single-task data)

    def __post_init__(self):
        if self.task is None:
            self.task = np.full(len(self.qids), "qa", dtype=object)

    def subset(self, mask):
        return Grid(self.qids[mask], self.dataset[mask], self.hops[mask], self.qtype[mask],
                    self.Q[mask], self.C[mask], self.config_ids, self.task[mask])


def realise(grid: Grid, pick: np.ndarray, overhead: np.ndarray | None = None):
    rows = np.arange(len(pick))
    q = grid.Q[rows, pick]
    c = grid.C[rows, pick] + (0 if overhead is None else overhead)
    return q, c


# ---------------------------------------------------------------- static baselines (fit on train)
def static_scores(train: Grid, lam: float, scale: float) -> np.ndarray:
    return train.Q.mean(axis=0) - lam * train.C.mean(axis=0) / scale


def static_pick(train: Grid, n_eval: int, lam: float = 0.0, scale: float = 1.0) -> np.ndarray:
    s = static_scores(train, lam, scale)
    c = choose(s[None, :], train.C.mean(axis=0)[None, :])[0]
    return np.full(n_eval, c)


def static_budget_pick(train: Grid, n_eval: int, budget: float) -> np.ndarray | None:
    """Best train-quality static config whose train mean cost <= budget (None if nothing fits)."""
    mq, mc = train.Q.mean(axis=0), train.C.mean(axis=0)
    ok = np.flatnonzero(mc <= budget + 1e-9)
    if len(ok) == 0:
        return None
    best = ok[choose(mq[ok][None, :], mc[ok][None, :])[0]]
    return np.full(n_eval, best)


def cheapest_pick(train: Grid, n_eval: int) -> np.ndarray:
    return np.full(n_eval, int(np.argmin(train.C.mean(axis=0))))


def per_group_static_pick(train: Grid, train_groups, eval_groups, lam: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Best static config per group (e.g. per dataset, or per predicted complexity class), fit on train.
    Groups unseen in train fall back to the global static choice."""
    fallback = static_pick(train, 1, lam, scale)[0]
    choice = {}
    for g in np.unique(train_groups):
        sub = train.subset(np.asarray(train_groups) == g)
        choice[g] = static_pick(sub, 1, lam, scale)[0]
    return np.array([choice.get(g, fallback) for g in eval_groups])


def random_expected(grid: Grid):
    """Uniform random router: exact per-query expectation over the configs (no sampling noise)."""
    return grid.Q.mean(axis=1), grid.C.mean(axis=1)


def random_sampled(grid: Grid, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, grid.Q.shape[1], size=grid.Q.shape[0])


# ---------------------------------------------------------------- learned router
def predicted_cost(train: Grid, eval_grid: Grid) -> np.ndarray:
    """Cost estimate per (query, config) = train mean cost of that config within the query's dataset.
    Cost is nearly deterministic given config + dataset (passages are fixed-length); its within-config
    coefficient of variation is reported to justify this."""
    out = np.empty_like(eval_grid.C)
    glob = train.C.mean(axis=0)
    for ds in np.unique(eval_grid.dataset):
        m_tr = train.dataset == ds
        out[eval_grid.dataset == ds] = train.C[m_tr].mean(axis=0) if m_tr.any() else glob
    return out


def router_pick(Qhat: np.ndarray, Chat: np.ndarray, lam: float, scale: float) -> np.ndarray:
    return choose(Qhat - lam * Chat / scale, Chat)


# ---------------------------------------------------------------- curves
def pareto_frontier(cost: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points (lower cost, higher quality), sorted by cost."""
    order = np.lexsort((-np.asarray(quality), np.asarray(cost)))
    keep, best = [], -np.inf
    for i in order:
        if quality[i] > best + 1e-12:
            keep.append(i)
            best = quality[i]
    return np.array(keep, dtype=int)


def quality_at_cost(cost: np.ndarray, quality: np.ndarray, budget: float) -> float:
    """Best quality among points with cost <= budget (step function; no interpolation)."""
    ok = np.asarray(cost) <= budget + 1e-9
    return float(np.max(np.asarray(quality)[ok])) if ok.any() else float("nan")


def cost_at_quality(cost: np.ndarray, quality: np.ndarray, target: float) -> float:
    ok = np.asarray(quality) >= target - 1e-12
    return float(np.min(np.asarray(cost)[ok])) if ok.any() else float("nan")


def select_lambda(lams, dev_q, dev_c, ref_q, ref_c, mode: str) -> float:
    """Pick lambda on DEV. iso_cost: max quality with cost <= ref_c. iso_quality: min cost with quality >= ref_q."""
    lams, dev_q, dev_c = map(np.asarray, (lams, dev_q, dev_c))
    if mode == "iso_cost":
        ok = dev_c <= ref_c + 1e-9
        if not ok.any():
            return float(lams[np.argmin(dev_c)])
        idx = np.flatnonzero(ok)
        return float(lams[idx[np.argmax(dev_q[idx])]])
    if mode == "iso_quality":
        ok = dev_q >= ref_q - 1e-12
        if not ok.any():
            return float(lams[np.argmax(dev_q)])
        idx = np.flatnonzero(ok)
        return float(lams[idx[np.argmin(dev_c[idx])]])
    raise ValueError(mode)
