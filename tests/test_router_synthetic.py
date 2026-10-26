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

    runs = pd.read_parquet(tmp_path / "synthetic_run" / "results" / "runs_scored.parquet")
    assert len(tables.static_grid(runs)) == 24 * 3
    assert len(tables.weight_sensitivity(runs, load_cost_weights())) > 4
    assert not tables.best_by_qtype(runs).empty
    from ragroute.oracle import oracle_null
    nul = oracle_null(runs, n_perm=20)
    # synthetic outcomes are noisy Bernoulli draws: the null oracle is already far above best static
    assert nul["oracle_null_mean"] > nul["best_static_on_split"]
    out = tmp_path / "figs"
    out.mkdir()
    plots.fig_pareto(curves, out, "router_A_gbdt", "router_B_gbdt")
    plots.fig_oracle_winrate(runs, out)
    plots.fig_ablation(headline, out)
    assert (out / "F2_pareto_test.png").exists()
