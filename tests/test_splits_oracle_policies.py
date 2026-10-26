import numpy as np
import pandas as pd

from ragroute.config import RagConfig, all_configs, load_experiment
from ragroute.cost import router_overhead
from ragroute.data.splits import grouped_split
from ragroute.oracle import oracle_best, oracle_cheapest_ok
from ragroute.router.policies import Grid, pareto_frontier, quality_at_cost, select_lambda, static_pick


def test_grid_has_24_unique_configs_and_roundtrip_ids():
    cfgs = all_configs(load_experiment())
    assert len(cfgs) == 24 and len({c.config_id for c in cfgs}) == 24
    for c in cfgs:
        assert RagConfig.from_id(c.config_id) == c


def test_grouped_split_no_shared_nonhub_titles():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(600):
        ds = ["nq", "hotpotqa"][i % 2]
        titles = [f"T{rng.integers(0, 1500)}", f"T{rng.integers(0, 1500)}"] if ds == "hotpotqa" else [f"S{i}"]
        if i % 50 == 0:
            titles.append("United States")  # hub
        rows.append({"qid": f"{ds}:{i}", "dataset": ds, "query": f"question number {i}", "gold_titles": titles})
    meta = pd.DataFrame(rows)
    out, rep = grouped_split(meta, {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=1, hub_max=10)
    assert rep["test_share_nonhub_title_with_train"] == 0.0
    assert rep["n_hub_titles"] >= 1
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.6) < 0.1


def test_grouped_split_refuses_giant_component():
    import pytest

    meta = pd.DataFrame({"qid": [f"q{i}" for i in range(100)], "dataset": "hotpotqa",
                         "query": [f"q {i}" for i in range(100)], "gold_titles": [["A", f"T{i}"] for i in range(100)]})
    with pytest.raises(ValueError):
        grouped_split(meta, {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=0, hub_max=1000)


def test_oracle_ties_go_to_cheapest():
    Q = np.array([[1.0, 1.0, 0.0], [0.0, 0.5, 0.5]])
    C = np.array([[5.0, 1.0, 0.1], [1.0, 3.0, 2.0]])
    assert list(oracle_best(Q, C)) == [1, 2]
    assert list(oracle_cheapest_ok(Q, C, tau=1.0)) == [1, 2]


def test_pareto_and_budget_helpers():
    cost = np.array([1, 2, 3, 4.0])
    q = np.array([0.3, 0.2, 0.5, 0.5])
    assert list(pareto_frontier(cost, q)) == [0, 2]
    assert quality_at_cost(cost, q, 2.5) == 0.3
    lam = select_lambda([0, 1, 2], [0.6, 0.5, 0.4], [10, 5, 3], ref_q=0.45, ref_c=6, mode="iso_cost")
    assert lam == 1


def test_static_pick_uses_train_only():
    tr = Grid(np.array(["a", "b"]), np.array(["nq"] * 2), np.array(["single"] * 2), np.array(["s"] * 2),
              np.array([[1.0, 0.0], [1.0, 0.0]]), np.ones((2, 2)), ["c0", "c1"])
    assert list(static_pick(tr, 3)) == [0, 0, 0]


def test_router_b_overhead_reuse_accounting():
    w = {"d": 2.0}
    ret = np.array(["bm25", "dense", "hybrid"])
    assert list(router_overhead("B", ret, w, reuse=True)) == [2.0, 2.0, 0.0]
    assert list(router_overhead("B", ret, w, reuse=False)) == [4.0, 4.0, 4.0]
    assert list(router_overhead("A", ret, w, reuse=True)) == [2.0, 0.0, 0.0]
