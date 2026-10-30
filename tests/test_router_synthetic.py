"""End-to-end dry run of router + tables + plots on SYNTHETIC data (never reported)."""
import pandas as pd

from ragroute.analysis import plots, tables
from ragroute.config import load_cost_weights
from ragroute.router.train_eval import run
from ragroute.synthetic import make


def test_router_learns_synthetic_structure(tmp_path):
    cfg = make(tmp_path / "synthetic_run", n_per_ds=150, seed=0)
    curves, picks, headline = run(cfg, models=["lr", "gbdt"], feature_sets=["A", "B", "D_only"], seeds=[0, 1])
    assert {"oracle_lambda", "static_lambda", "adaptive_rag_style", "router_B_gbdt"} <= set(curves["policy"])
    t = curves[(curves["split"] == "test") & (curves["group"] == "ALL")]
    oracle = t[t["policy"] == "oracle_best"]["quality"].iloc[0]
    static0 = t[(t["policy"] == "static_lambda") & (t["lambda"] == 0)]["quality"].iloc[0]
    rb0 = t[(t["policy"] == "router_B_gbdt") & (t["lambda"] == 0)]["quality"].mean()
    assert oracle >= rb0 - 1e-9 and oracle > static0
    assert set(headline["mode"]) == {"iso_cost", "iso_quality", "lambda0"}

    # global + per-task routers: synthetic data has 3 tasks (QA x3 datasets, FEVER, zsRE)
    tasks = {"qa", "verification", "slot_filling"}
    assert set(headline["eval_on"]) == {"ALL"} | {f"task={t}" for t in tasks}
    allq = headline[headline["eval_on"] == "ALL"]
    assert {"global", "per_task"} == set(allq["scope"])
    assert "router_B_gbdt@per_task" in set(allq["policy"])
    for t in tasks:  # each task block holds the global router and that task's own router
        blk = headline[headline["eval_on"] == f"task={t}"]
        assert {"router_B_gbdt", f"router_B_gbdt@{t}"} <= set(blk["policy"])
        assert blk["n"].nunique() == 1 and blk["n"].iloc[0] < allq["n"].iloc[0]
    assert headline["dQ_p_holm"].between(0, 1).all() and (headline["dQ_p_holm"] >= headline["dQ_p"] - 1e-12).all()
    assert allq["dQ_macro_task"].notna().all()
    assert "per_task_static" in set(curves["policy"])
    assert {f"task={t}" for t in tasks} <= set(curves["group"])

    runs = pd.read_parquet(tmp_path / "synthetic_run" / "results" / "runs_scored.parquet")
    assert len(tables.static_grid(runs)) == 24 * runs["dataset"].nunique()
    assert len(tables.weight_sensitivity(runs, load_cost_weights())) > 4
    assert not tables.best_by_qtype(runs).empty
    out = tmp_path / "figs"
    out.mkdir()
    plots.fig_pareto(curves, out, "router_A_gbdt", "router_B_gbdt")
    plots.fig_oracle_winrate(runs, out)
    plots.fig_ablation(headline, out)
    assert (out / "F2_pareto_test.png").exists() and (out / "F3b_pareto_by_task.png").exists()
