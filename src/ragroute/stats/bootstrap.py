"""Paired bootstrap over shared queries.

Every config-vs-config and router-vs-baseline claim in the report goes through `paired_bootstrap`:
both systems are evaluated on the same queries, we resample query indices with replacement
(B = 10,000), and recompute the mean difference on each resample.

- CI: percentile interval of the resampled differences.
- p-value: two-sided, null-centred: p = (1 + #{|d*_b - d_obs| >= |d_obs|}) / (1 + B).
- strata: optional per-query labels (e.g. dataset). Resampling is done within each stratum so every
  resample keeps the dataset mix of the test set. statistic="macro" averages stratum means equally.

Routers trained with several seeds: average each query's outcome over seeds first, then bootstrap
over queries; report seed spread separately (seed variance is not query-sampling variance).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BootResult:
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_resamples: int
    statistic: str

    def as_dict(self) -> dict:
        return asdict(self)


def _resample_index_blocks(n_by_stratum, rng, n_resamples, chunk):
    """Yields (chunk_size, n_total) arrays of resampled indices, stratified."""
    offsets = np.cumsum([0] + [len(ix) for ix in n_by_stratum])
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        parts = [ix[rng.integers(0, len(ix), size=(m, len(ix)))] for ix in n_by_stratum]
        yield np.concatenate(parts, axis=1), offsets
        done += m


def _stat(values_resampled, offsets, statistic):
    if statistic == "micro":
        return values_resampled.mean(axis=1)
    # macro: equal weight per stratum
    means = [values_resampled[:, offsets[i]:offsets[i + 1]].mean(axis=1) for i in range(len(offsets) - 1)]
    return np.mean(means, axis=0)


def paired_bootstrap(
    a,
    b,
    n_resamples: int = 10_000,
    seed: int = 0,
    strata=None,
    statistic: str = "micro",
    ci: float = 0.95,
    chunk: int = 500,
) -> BootResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-d arrays over the same queries")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN in per-query scores; drop or impute explicitly before testing")
    d = a - b
    strata = np.zeros(len(d), dtype=int) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    # observed statistic on the original sample
    order = np.concatenate(groups)
    offsets0 = np.cumsum([0] + [len(g) for g in groups])
    obs = float(_stat(d[order][None, :], offsets0, statistic)[0])
    mean_a = float(_stat(a[order][None, :], offsets0, statistic)[0])
    mean_b = float(_stat(b[order][None, :], offsets0, statistic)[0])

    rng = np.random.default_rng(seed)
    boots = []
    for idx, offsets in _resample_index_blocks(groups, rng, n_resamples, chunk):
        boots.append(_stat(d[idx], offsets, statistic))
    boots = np.concatenate(boots)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    p = (1 + np.sum(np.abs(boots - obs) >= abs(obs))) / (1 + n_resamples)
    return BootResult(mean_a, mean_b, obs, float(lo), float(hi), float(min(p, 1.0)), len(d), n_resamples, statistic)


def bootstrap_mean_ci(x, n_resamples: int = 10_000, seed: int = 0, strata=None, statistic="micro", ci=0.95):
    """CI for a single system's mean (paired against a zero vector)."""
    x = np.asarray(x, dtype=float)
    r = paired_bootstrap(x, np.zeros_like(x), n_resamples, seed, strata, statistic, ci)
    return {"mean": r.mean_a, "ci_low": r.ci_low, "ci_high": r.ci_high, "n": r.n}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control for a set of comparisons)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def best_config_stability(Q, config_ids, n_resamples: int = 10_000, seed: int = 0, strata=None, chunk: int = 200):
    """How stable is 'config X is best' for a subset of queries (e.g. one question type)?

    Q: (n_queries, n_configs) per-query quality matrix.
    Returns per-config frequency of being the argmax across resamples, the observed best, and a paired
    bootstrap CI for (best - runner_up). A per-question-type 'best config' claim is only made when the
    CI for the margin excludes 0.
    """
    Q = np.asarray(Q, dtype=float)
    n, c = Q.shape
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    wins = np.zeros(c)
    for idx, _ in _resample_index_blocks(groups, rng, n_resamples, chunk):
        means = Q[idx].mean(axis=1)            # (m, c)
        np.add.at(wins, means.argmax(axis=1), 1)
    col_means = Q.mean(axis=0)
    order = np.argsort(-col_means)
    best, runner = order[0], order[1]
    margin = paired_bootstrap(Q[:, best], Q[:, runner], n_resamples, seed + 1, strata)
    return {
        "best": config_ids[best],
        "runner_up": config_ids[runner],
        "win_freq": dict(zip(config_ids, (wins / n_resamples).round(4))),
        "margin": margin.as_dict(),
        "claim_supported": margin.ci_low > 0,
    }
