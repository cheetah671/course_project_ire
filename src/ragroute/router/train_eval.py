"""Train and evaluate routers, baselines and the oracle; write every number the report needs.

Formulation (recommended): one pooled model predicts quality q_hat(query, config) for all 24 configs;
cost c_hat(query, config) = train mean cost of the config within the dataset; route by
    argmax_c  q_hat - lambda * c_hat / scale
Sweeping lambda traces the router's cost-quality curve without retraining.

Feature groups -> feature sets (ablations):
    query  = surface/cue features + PCA(question embedding) + learned multi-hop score   (group 1)
    data   = dataset one-hots + dataset-level attributes                               (group 2)
    retrieval = Router B probe signals                                                 (post-retrieval)
    config = encoded config (group 3) is always part of the pair features.
    Group 4 (observed quality / cost of the pair) is the TRAINING TARGET, never an input.

Outputs ({results}/router/; with --no-reuse: {results}/router_noreuse/):
    curves.parquet     policy curves on dev and test (overall, per dataset, per hops)
    picks.parquet      per-query test picks at the selected lambdas (input to bootstrap/plots)
    headline.csv       iso-cost and iso-quality comparisons vs best static, with paired bootstrap CIs
    importance.json    GBDT gain aggregated by feature group
    static_points.parquet  the 24 static configs on test (for the Pareto plot)

Usage: python -m ragroute.router.train_eval [--models lr gbdt mlp] [--feature-sets A B ...] [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from ragroute.config import RagConfig, config_feature_frame, load_cost_weights, load_experiment
from ragroute.cost import router_overhead
from ragroute.oracle import cost_scale, oracle_best, oracle_lambda, wide
from ragroute.router.models import make_model
from ragroute.router.policies import (Grid, cheapest_pick, per_group_static_pick, predicted_cost, quality_at_cost,
                                      random_expected, realise, router_pick, select_lambda, static_pick)
from ragroute.stats.bootstrap import paired_bootstrap

FEATURE_SETS = {
    "A_noDS": ["query"],
    "A": ["query", "data"],
    "A_noEmb": ["query_surface", "data"],
    "B": ["query", "data", "retrieval"],
    "B_noDS": ["query", "retrieval"],
    "D_only": ["data"],
    "R_only": ["retrieval"],
}
LAMBDAS = np.r_[0.0, np.geomspace(1e-3, 5.0, 60)]
N_PCA = 32


# ------------------------------------------------------------------ data assembly
def build_grid(runs: pd.DataFrame, meta: pd.DataFrame, split: str) -> Grid:
    sub = runs[runs["split"] == split]
    Qd, Cd = wide(sub, "quality"), wide(sub, "cost")
    m = meta.set_index("qid").loc[Qd.index]
    return Grid(Qd.index.to_numpy(), m["dataset"].to_numpy(), m["hops"].to_numpy(), m["qtype"].to_numpy(),
                Qd.to_numpy(), Cd.to_numpy(), list(Qd.columns))


def query_frame(qf: pd.DataFrame, emb: np.ndarray, rf: pd.DataFrame, meta: pd.DataFrame, train_qids, seed: int):
    """All query-level features, with PCA and the multi-hop score fitted on TRAIN queries only."""
    qf = qf.set_index("qid")
    emb_df = pd.DataFrame(emb, index=qf.index)
    tr = qf.index.isin(set(train_qids))
    pca = PCA(n_components=N_PCA, random_state=seed).fit(emb_df[tr])
    pcs = pd.DataFrame(pca.transform(emb_df), index=qf.index, columns=[f"emb_pc{i}" for i in range(N_PCA)])
    surface_cols = [c for c in qf.columns if c.startswith("q_")]
    data_cols = [c for c in qf.columns if c.startswith("ds_")]
    X_mh = pd.concat([qf[surface_cols], pcs], axis=1)
    y_mh = (meta.set_index("qid").loc[qf.index, "hops"] == "multi").astype(int)
    mh = pd.Series(np.nan, index=qf.index)
    if y_mh[tr].nunique() > 1:
        clf = lambda: LogisticRegression(C=0.5, max_iter=2000)  # noqa: E731
        sc = StandardScaler().fit(X_mh[tr])
        mh[tr] = cross_val_predict(clf(), sc.transform(X_mh[tr]), y_mh[tr], cv=5, method="predict_proba")[:, 1]
        mh[~tr] = clf().fit(sc.transform(X_mh[tr]), y_mh[tr]).predict_proba(sc.transform(X_mh[~tr]))[:, 1]
    else:
        mh[:] = 0.0
    frame = pd.concat([qf[surface_cols], pcs, mh.rename("q_multihop_score"), qf[data_cols],
                       rf.set_index("qid").loc[qf.index]], axis=1)
    groups = {
        "query_surface": surface_cols,
        "query": surface_cols + list(pcs.columns) + ["q_multihop_score"],
        "data": data_cols,
        "retrieval": list(rf.columns.drop("qid")),
    }
    return frame, groups


def pair_matrix(Fq: np.ndarray, Fc: np.ndarray, interactions: bool) -> np.ndarray:
    """Rows are query-major: row i*C + j = (query i, config j)."""
    n, C = len(Fq), len(Fc)
    q_rep = np.repeat(Fq, C, axis=0)
    c_rep = np.tile(Fc, (n, 1))
    parts = [q_rep, c_rep]
    if interactions:
        parts.append((q_rep[:, :, None] * c_rep[:, None, :]).reshape(n * C, -1))
    return np.hstack(parts)


def overhead_kind(groups: list[str]) -> str:
    if "retrieval" in groups:
        return "B"
    if "query" in groups:
        return "A"  # needs the question embedding
    return "none"   # surface features / dataset id only: negligible cost


# ------------------------------------------------------------------ main experiment
def run(cfg: dict, models: list[str], feature_sets: list[str], seeds: list[int], reuse: bool = True):
    res = Path(cfg["paths"]["results_dir"])
    # the conservative (no probe reuse) run is a sensitivity check: keep it apart so it never overwrites the main run
    out = res / ("router" if reuse else "router_noreuse")
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(res / "runs_scored.parquet")
    meta = pd.read_parquet(Path(cfg["paths"]["data_root"]) / "qa_meta.parquet")
    feats = res / "features"
    qf = pd.read_parquet(feats / "query_features.parquet")
    emb = np.load(feats / "question_embeddings.npy")
    rf = pd.read_parquet(feats / "retrieval_features.parquet")
    weights = load_cost_weights()
    w = weights["weight_sets"][weights["default"]]

    grids = {s: build_grid(runs, meta, s) for s in ("train", "dev", "test")}
    tr, dv, te = grids["train"], grids["dev"], grids["test"]
    configs = tr.config_ids
    retr = np.array([RagConfig.from_id(c).retriever for c in configs])
    Fc_df = config_feature_frame([RagConfig.from_id(c) for c in configs]).set_index("config_id").loc[configs]
    Fc = Fc_df.to_numpy(float)
    scale = cost_scale(runs)
    curves, picks, importance = [], [], {}

    def record(policy, split, lam, q, c, grid, extra=None):
        base = {"policy": policy, "split": split, "lambda": lam, **(extra or {})}
        curves.append({**base, "group": "ALL", "quality": q.mean(), "cost": c.mean(), "n": len(q)})
        for key, labels in (("dataset", grid.dataset), ("hops", grid.hops)):
            for g in np.unique(labels):
                m = labels == g
                curves.append({**base, "group": f"{key}={g}", "quality": q[m].mean(), "cost": c[m].mean(), "n": int(m.sum())})

    # ---------- baselines and oracle
    for split, g in (("dev", dv), ("test", te)):
        for lam in LAMBDAS:
            q, c = realise(g, static_pick(tr, len(g.qids), lam, scale)); record("static_lambda", split, lam, q, c, g)
            q, c = realise(g, per_group_static_pick(tr, tr.dataset, g.dataset, lam, scale))
            record("per_dataset_static", split, lam, q, c, g)
            q, c = realise(g, oracle_lambda(g.Q, g.C, lam, scale)); record("oracle_lambda", split, lam, q, c, g)
        q, c = realise(g, oracle_best(g.Q, g.C)); record("oracle_best", split, np.nan, q, c, g)
        q, c = realise(g, cheapest_pick(tr, len(g.qids))); record("cheapest", split, np.nan, q, c, g)
        q, c = random_expected(g); record("random", split, np.nan, q, c, g)
        for j, cid in enumerate(configs):
            q, c = g.Q[:, j], g.C[:, j]; record("static_config", split, np.nan, q, c, g, {"config_id": cid})

    # Adaptive-RAG-style: complexity classifier on question text -> best static per predicted class
    frame0, groups0 = query_frame(qf, emb, rf, meta, tr.qids, seed=0)
    cls_cols = groups0["query_surface"] + [c for c in groups0["query"] if c.startswith("emb_pc")]
    label = lambda g: np.where(np.isin(g.dataset, ["fever"]), "verification", g.hops)  # noqa: E731
    sc = StandardScaler().fit(frame0.loc[tr.qids, cls_cols])
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(frame0.loc[tr.qids, cls_cols]), label(tr))
    for split, g in (("dev", dv), ("test", te)):
        pred = clf.predict(sc.transform(frame0.loc[g.qids, cls_cols]))
        for lam in LAMBDAS:
            pick = per_group_static_pick(tr, label(tr), pred, lam, scale)
            q, c = realise(g, pick, router_overhead("A", retr[pick], w, reuse))
            record("adaptive_rag_style", split, lam, q, c, g, {"clf_acc": float((pred == label(g)).mean())})

    # ---------- learned routers
    for fs in feature_sets:
        groups = FEATURE_SETS[fs]
        kind = overhead_kind(groups)
        for model_name in models:
            for seed in seeds:
                frame, gcols = query_frame(qf, emb, rf, meta, tr.qids, seed)
                cols = [c for g in groups for c in gcols[g]]
                model = make_model(model_name, seed)
                inter = getattr(model, "needs_interactions", False)
                Xtr = pair_matrix(frame.loc[tr.qids, cols].to_numpy(float), Fc, inter)
                model.fit(Xtr, tr.Q.reshape(-1))
                if model_name == "gbdt" and seed == seeds[0] and hasattr(model, "feature_importance"):
                    imp = model.feature_importance()
                    if imp is not None:
                        names = cols + list(Fc_df.columns)
                        by_group = {g: float(sum(imp[names.index(c)] for c in gcols[g] if c in names)) for g in groups}
                        by_group["config"] = float(sum(imp[len(cols):len(cols) + Fc.shape[1]]))
                        importance[fs] = by_group
                pol = f"router_{fs}_{model_name}"
                sel = {}
                for split, g in (("dev", dv), ("test", te)):
                    Qhat = model.predict(pair_matrix(frame.loc[g.qids, cols].to_numpy(float), Fc, inter)).reshape(len(g.qids), -1)
                    Chat = predicted_cost(tr, g)
                    lam_q, lam_c = [], []
                    for lam in LAMBDAS:
                        pick = router_pick(Qhat, Chat, lam, scale)
                        q, c = realise(g, pick, router_overhead(kind, retr[pick], w, reuse))
                        record(pol, split, lam, q, c, g, {"seed": seed, "feature_set": fs, "model": model_name})
                        lam_q.append(q.mean()); lam_c.append(c.mean())
                    if split == "dev":
                        ref = static_pick(tr, len(g.qids), 0.0, scale)
                        rq, rc = realise(g, ref)
                        sel["iso_cost"] = select_lambda(LAMBDAS, lam_q, lam_c, rq.mean(), rc.mean(), "iso_cost")
                        sel["iso_quality"] = select_lambda(LAMBDAS, lam_q, lam_c, rq.mean(), rc.mean(), "iso_quality")
                        sel["lambda0"] = 0.0
                    else:
                        for mode, lam in sel.items():
                            pick = router_pick(Qhat, Chat, lam, scale)
                            q, c = realise(g, pick, router_overhead(kind, retr[pick], w, reuse))
                            picks.append(pd.DataFrame({"policy": pol, "mode": mode, "lambda": lam, "seed": seed,
                                                       "qid": g.qids, "config_id": np.array(configs)[pick],
                                                       "quality": q, "cost": c}))
                print(f"done {pol} seed={seed} selected={sel}")

    curves_df = pd.DataFrame(curves)
    curves_df.to_parquet(out / "curves.parquet", index=False)
    picks_df = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    picks_df.to_parquet(out / "picks.parquet", index=False)
    (out / "importance.json").write_text(json.dumps(importance, indent=2))
    headline = headline_table(picks_df, te, tr, scale, curves_df)
    headline.to_csv(out / "headline.csv", index=False)
    print(headline.to_string())
    return curves_df, picks_df, headline


def headline_table(picks: pd.DataFrame, te: Grid, tr: Grid, scale: float, curves: pd.DataFrame) -> pd.DataFrame:
    """Router (seed-averaged per query) vs best static (chosen on train), paired bootstrap stratified by dataset."""
    ref_pick = static_pick(tr, len(te.qids), 0.0, scale)
    ref_q, ref_c = realise(te, ref_pick)
    oracle = curves[(curves["policy"] == "oracle_lambda") & (curves["split"] == "test") & (curves["group"] == "ALL")]
    rows = []
    for (pol, mode), sub in picks.groupby(["policy", "mode"]):
        per_q = sub.groupby("qid")[["quality", "cost"]].mean().loc[te.qids]
        bq = paired_bootstrap(per_q["quality"].to_numpy(), ref_q, strata=te.dataset)
        bc = paired_bootstrap(per_q["cost"].to_numpy(), ref_c, strata=te.dataset)
        q_or = quality_at_cost(oracle["cost"].to_numpy(), oracle["quality"].to_numpy(), ref_c.mean())
        gap = (bq.mean_a - bq.mean_b) / (q_or - bq.mean_b) if q_or - bq.mean_b > 1e-9 else float("nan")
        rows.append({"policy": pol, "mode": mode, "n_seeds": sub["seed"].nunique(),
                     "router_quality": bq.mean_a, "static_quality": bq.mean_b, "dQ": bq.diff,
                     "dQ_ci_low": bq.ci_low, "dQ_ci_high": bq.ci_high, "dQ_p": bq.p_value,
                     "router_cost": bc.mean_a, "static_cost": bc.mean_b,
                     "cost_saving_pct": 100 * (1 - bc.mean_a / bc.mean_b),
                     "dC_ci_low": bc.ci_low, "dC_ci_high": bc.ci_high,
                     "oracle_quality_at_static_cost": q_or, "gap_closed": gap,
                     "seed_sd_quality": float(sub.groupby("seed")["quality"].mean().std(ddof=1)) if sub["seed"].nunique() > 1 else 0.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["lr", "gbdt", "mlp"])
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--no-reuse", action="store_true", help="conservative Router B accounting (probe never reused)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    run(cfg, args.models, args.feature_sets, args.seeds or cfg["router_seeds"], reuse=not args.no_reuse)


if __name__ == "__main__":
    main()
