"""Tasks 2/3 (FEVER, zsRE), the per-task router combination, and cost-weight calibration."""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ragroute.config import load_cost_weights, load_experiment
from ragroute.data.prepare_qa import zsre_query
from ragroute.eval.metrics import score_row
from ragroute.router.policies import Grid
from ragroute.router.train_eval import add_combined_per_task

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_costs", REPO / "scripts" / "calibrate_costs.py")
calibrate_costs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_costs)


def test_three_tasks_enabled_with_prompts():
    cfg = load_experiment()
    tasks = {d["task"] for d in cfg["datasets"].values() if d.get("enabled")}
    assert {"qa", "verification", "slot_filling"} <= tasks
    assert tasks <= set(cfg["prompts"])


def test_zsre_query_and_slot_filling_metric():
    assert zsre_query("USA-88 [SEP] time of spacecraft launch") == "USA-88 | time of spacecraft launch"
    r = score_row("slot_filling", "Value: 3 February 1993", ["3 February 1993"])
    assert r["quality"] == r["em"] == 1.0
    partial = score_row("slot_filling", "February 1993", ["3 February 1993"])
    assert partial["quality"] == 0.0 and 0 < partial["f1"] < 1  # EM is the target; F1 only reported


def test_grid_default_task_and_subset():
    g = Grid(np.array(["a", "b"]), np.array(["nq", "fever"]), np.array(["single"] * 2), np.array(["s"] * 2),
             np.ones((2, 2)), np.ones((2, 2)), ["c0", "c1"], np.array(["qa", "verification"]))
    assert list(g.subset(g.task == "verification").qids) == ["b"]
    assert list(Grid(g.qids, g.dataset, g.hops, g.qtype, g.Q, g.C, g.config_ids).task) == ["qa", "qa"]


def test_combined_per_task_router_covers_all_queries():
    rows = [{"policy": f"router_B_gbdt@{t}", "scope": t, "mode": "iso_cost", "seed": 0, "qid": q, "quality": 1.0, "cost": 1.0}
            for t, qs in (("qa", ["q1", "q2"]), ("verification", ["q3"])) for q in qs]
    out = add_combined_per_task(pd.DataFrame(rows), ["qa", "verification"], n_test=3)
    comb = out[out["scope"] == "per_task"]
    assert set(comb["policy"]) == {"router_B_gbdt@per_task"} and comb["qid"].nunique() == 3
    with pytest.raises(ValueError):
        add_combined_per_task(pd.DataFrame(rows), ["qa", "verification"], n_test=4)


def _fake_timings(a_in, a_out, b):
    rows = []
    for k, tin in ((1, 230), (3, 520), (10, 1550)):
        for m in (1, 32):
            n = 200
            secs = (a_in * tin * n + a_out * m * n + b * n) / 1000.0
            rows.append({"component": "generate", "generator": "g", "k": k, "max_tokens": m, "n_calls": n,
                         "input_tokens": tin * n, "output_tokens": m * n, "seconds": secs, "gpus": 1})
    return rows


def test_calibration_recovers_known_weights():
    cal = _fake_timings(0.3, 12.0, 4.0) + [{"component": "rerank", "pairs": 4000, "seconds": 2.0, "gpus": 1}]
    stage = [{"stage": "retrieval_bm25", "sec_per_query_autorag": 0.004},
             {"stage": "retrieval_dense", "sec_per_query_autorag": 0.002}]
    w, rep = calibrate_costs.fit_weights(cal, stage)
    assert w["a_in"]["g"] == pytest.approx(0.3, rel=1e-6) and w["a_out"]["g"] == pytest.approx(12.0, rel=1e-6)
    assert w["b"] == pytest.approx(4.0, rel=1e-4)
    assert w["c"] == pytest.approx(0.5) and w["d"] == pytest.approx(3.0)
    assert rep["generation_fit"]["g"]["r2"] > 0.999 and rep["rerank_source"] == "calibration"


def test_cost_weights_measured_overrides_placeholder(tmp_path):
    missing = tmp_path / "none.yaml"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = load_cost_weights(measured_path=missing)
    assert w["default"] == "measured_placeholder" and any("PLACEHOLDER" in str(c.message) for c in caught)
    measured = tmp_path / "m.yaml"
    measured.write_text(yaml.safe_dump({"weights": {"a_in": {"qwen7b": 1, "qwen1p5b": 1}, "a_out": {"qwen7b": 1, "qwen1p5b": 1},
                                                    "b": 0, "c": 0, "d": 0}}))
    w = load_cost_weights(measured_path=measured)
    assert w["default"] == "measured" and "measured" in w["weight_sets"]
