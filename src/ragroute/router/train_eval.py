"""Train and evaluate routers, baselines and the oracle; write every number the report needs.

Formulation (recommended): one pooled model predicts quality q_hat(query, config) for all 24 configs;
cost c_hat(query, config) = train mean cost of the config within the dataset; route by
    argmax_c  q_hat - lambda * c_hat / scale
Sweeping lambda traces the router's cost-quality curve without retraining.

Global vs per-task routers (the professor asked for >= 2 tasks: QA, verification, slot filling):
    global     one router trained on all tasks pooled (task identity enters only through the data features)
    per task   one router per task, trained, lambda-selected and evaluated on that task's queries only
               (policy name suffix "@<task>"); "@per_task" = the per-task routers combined, so every test
               query is routed by its own task's router (comparable to the global router on ALL queries).
    headline.csv has one block per evaluation set (eval_on = ALL, task=qa, task=verification, ...). In a task
    block, the global router and that task's own router are both compared with the best static config chosen
    on THAT task's train split.

Feature groups -> feature sets (ablations):
    query  = surface/cue features + PCA(question embedding) + learned multi-hop score   (group 1)
    data   = dataset one-hots + dataset-level attributes                               (group 2)
    retrieval = Router B probe signals                                                 (post-retrieval)
    config = encoded config (group 3) is always part of the pair features.
    Group 4 (observed quality / cost of the pair) is the TRAINING TARGET, never an input.

Outputs ({results}/router/; with --no-reuse: {results}/router_noreuse/):
    curves.parquet     policy curves on dev and test (overall, per dataset, per hops)
    picks.parquet      per-query test picks at the selected lambdas (input to bootstrap/plots)
    headline.csv       iso-cost and iso-quality comparisons vs best static, with paired bootstrap CIs (micro, macro
                       over datasets and, with >1 task, macro over tasks) and Holm-adjusted p within each
                       (eval_on, mode) family
    importance.json    GBDT gain aggregated by feature group
    static_points.parquet  the 24 static configs on test (for the Pareto plot)

Usage: python -m ragroute.router.train_eval [--models lr gbdt mlp] [--feature-sets A B ...] [--no-per-task]
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
from ragroute.stats.bootstrap import holm, paired_bootstrap

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
GLOBAL = "global"
PER_TASK = "per_task"


# ------------------------------------------------------------------ data assembly
def build_grid(runs: pd.DataFrame, meta: pd.DataFrame, split: str) -> Grid:
    sub = runs[runs["split"] == split]
    Qd, Cd = wide(sub, "quality"), wide(sub, "cost")
    m = meta.set_index("qid").loc[Qd.index]
    return Grid(Qd.index.to_numpy(), m["dataset"].to_numpy(), m["hops"].to_numpy(), m["qtype"].to_numpy(),
                Qd.to_numpy(), Cd.to_numpy(), list(Qd.columns), m["task"].to_numpy())


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
def run(cfg: dict, models: list[str], feature_sets: list[str], seeds: list[int], reuse: bool = True,
        per_task: bool = True):
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
    tasks = sorted(np.unique(tr.task))
    multi_task = len(tasks) > 1
    curves, picks, importance = [], [], {}

    def record(policy, split, lam, q, c, grid, extra=None, all_label="ALL"):
        """all_label: name of the whole-grid group ("ALL", or "task=<t>" for a per-task router's own grid)."""
        base = {"policy": policy, "split": split, "lambda": lam, "scope": GLOBAL, **(extra or {})}
        curves.append({**base, "group": all_label, "quality": q.mean(), "cost": c.mean(), "n": len(q)})
        keys = [("dataset", grid.dataset), ("hops", grid.hops)]
        if multi_task and all_label == "ALL":
            keys.append(("task", grid.task))
        for key, labels in keys:
            for g in np.unique(labels):
                m = labels == g
                curves.append({**base, "group": f"{key}={g}", "quality": q[m].mean(), "cost": c[m].mean(), "n": int(m.sum())})

    # ---------- baselines and oracle
    for split, g in (("dev", dv), ("test", te)):
        for lam in LAMBDAS:
            q, c = realise(g, static_pick(tr, len(g.qids), lam, scale)); record("static_lambda", split, lam, q, c, g)
            q, c = realise(g, per_group_static_pick(tr, tr.dataset, g.dataset, lam, scale))
            record("per_dataset_static", split, lam, q, c, g)
            if multi_task:  # best static config per task (the static counterpart of the per-task routers)
                q, c = realise(g, per_group_static_pick(tr, tr.task, g.task, lam, scale))
                record("per_task_static", split, lam, q, c, g)
            q, c = realise(g, oracle_lambda(g.Q, g.C, lam, scale)); record("oracle_lambda", split, lam, q, c, g)
        q, c = realise(g, oracle_best(g.Q, g.C)); record("oracle_best", split, np.nan, q, c, g)
        q, c = realise(g, cheapest_pick(tr, len(g.qids))); record("cheapest", split, np.nan, q, c, g)
        q, c = random_expected(g); record("random", split, np.nan, q, c, g)
        for j, cid in enumerate(configs):
            q, c = g.Q[:, j], g.C[:, j]; record("static_config", split, np.nan, q, c, g, {"config_id": cid})

    # Adaptive-RAG-style: complexity classifier on question text -> best static per predicted class
    frame0, groups0 = query_frame(qf, emb, rf, meta, tr.qids, seed=0)
    cls_cols = groups0["query_surface"] + [c for c in groups0["query"] if c.startswith("emb_pc")]
    label = lambda g: np.where(g.task != "qa", g.task, g.hops)  # noqa: E731  classes: single / multi / <non-QA task>
    sc = StandardScaler().fit(frame0.loc[tr.qids, cls_cols])
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(frame0.loc[tr.qids, cls_cols]), label(tr))
    for split, g in (("dev", dv), ("test", te)):
        pred = clf.predict(sc.transform(frame0.loc[g.qids, cls_cols]))
        for lam in LAMBDAS:
            pick = per_group_static_pick(tr, label(tr), pred, lam, scale)
            q, c = realise(g, pick, router_overhead("A", retr[pick], w, reuse))
            record("adaptive_rag_style", split, lam, q, c, g, {"clf_acc": float((pred == label(g)).mean())})

    # ---------- learned routers: one global router, plus one router per task
    scopes = [(GLOBAL, None)] + ([(t, t) for t in tasks] if per_task and multi_task else [])
    for scope, task in scopes:
        if task is None:
            s_tr, s_dv, s_te, suffix, all_label = tr, dv, te, "", "ALL"
        else:
            s_tr, s_dv, s_te = (g.subset(g.task == task) for g in (tr, dv, te))
            suffix, all_label = f"@{task}", f"task={task}"
        for fs in feature_sets:
            groups = FEATURE_SETS[fs]
            kind = overhead_kind(groups)
            for model_name in models:
                for seed in seeds:
                    frame, gcols = query_frame(qf, emb, rf, meta, s_tr.qids, seed)
                    cols = [c for g in groups for c in gcols[g]]
                    model = make_model(model_name, seed)
                    inter = getattr(model, "needs_interactions", False)
                    Xtr = pair_matrix(frame.loc[s_tr.qids, cols].to_numpy(float), Fc, inter)
                    model.fit(Xtr, s_tr.Q.reshape(-1))
                    if model_name == "gbdt" and seed == seeds[0] and hasattr(model, "feature_importance"):
                        imp = model.feature_importance()
                        if imp is not None:
                            names = cols + list(Fc_df.columns)
                            by_group = {g: float(sum(imp[names.index(c)] for c in gcols[g] if c in names)) for g in groups}
                            by_group["config"] = float(sum(imp[len(cols):len(cols) + Fc.shape[1]]))
                            importance[fs + suffix] = by_group
                    pol = f"router_{fs}_{model_name}{suffix}"
                    info = {"seed": seed, "feature_set": fs, "model": model_name, "scope": scope}
                    sel = {}
                    for split, g in (("dev", s_dv), ("test", s_te)):
                        Qhat = model.predict(pair_matrix(frame.loc[g.qids, cols].to_numpy(float), Fc, inter)).reshape(len(g.qids), -1)
                        Chat = predicted_cost(s_tr, g)
                        lam_q, lam_c = [], []
                        for lam in LAMBDAS:
                            pick = router_pick(Qhat, Chat, lam, scale)
                            q, c = realise(g, pick, router_overhead(kind, retr[pick], w, reuse))
                            record(pol, split, lam, q, c, g, info, all_label)
                            lam_q.append(q.mean()); lam_c.append(c.mean())
                        if split == "dev":
                            # lambda is selected against the best static config of the SAME scope (task or all)
                            ref = static_pick(s_tr, len(g.qids), 0.0, scale)
                            rq, rc = realise(g, ref)
                            sel["iso_cost"] = select_lambda(LAMBDAS, lam_q, lam_c, rq.mean(), rc.mean(), "iso_cost")
                            sel["iso_quality"] = select_lambda(LAMBDAS, lam_q, lam_c, rq.mean(), rc.mean(), "iso_quality")
                            sel["lambda0"] = 0.0
                        else:
                            for mode, lam in sel.items():
                                pick = router_pick(Qhat, Chat, lam, scale)
                                q, c = realise(g, pick, router_overhead(kind, retr[pick], w, reuse))
                                picks.append(pd.DataFrame({"policy": pol, "scope": scope, "mode": mode, "lambda": lam,
                                                           "seed": seed, "qid": g.qids,
                                                           "config_id": np.array(configs)[pick],
                                                           "quality": q, "cost": c}))
                    print(f"done {pol} seed={seed} selected={sel}")

    curves_df = pd.DataFrame(curves)
    curves_df.to_parquet(out / "curves.parquet", index=False)
    picks_df = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    picks_df = add_combined_per_task(picks_df, tasks, len(te.qids))
    picks_df.to_parquet(out / "picks.parquet", index=False)
    (out / "importance.json").write_text(json.dumps(importance, indent=2))
    headline = headline_all(picks_df, te, tr, scale, curves_df, tasks)
    headline.to_csv(out / "headline.csv", index=False)
    print(headline.to_string())
    return curves_df, picks_df, headline


def add_combined_per_task(picks: pd.DataFrame, tasks: list, n_test: int) -> pd.DataFrame:
    """'router_<fs>_<model>@per_task': each test query routed by its own task's router (lambda chosen per task)."""
    if picks.empty or len(tasks) < 2:
        return picks
    pt = picks[picks["scope"].isin(tasks)]
    if pt.empty:
        return picks
    comb = pt.assign(policy=pt["policy"].str.replace(r"@[^@]+$", f"@{PER_TASK}", regex=True), scope=PER_TASK)
    cover = comb.groupby(["policy", "mode", "seed"])["qid"].nunique()
    if (cover != n_test).any():
        raise ValueError(f"per-task routers do not cover every test query: {cover[cover != n_test].head()}")
    return pd.concat([picks, comb], ignore_index=True)


def headline_all(picks: pd.DataFrame, te: Grid, tr: Grid, scale: float, curves: pd.DataFrame, tasks: list) -> pd.DataFrame:
    """Headline blocks: ALL test queries (global and combined per-task routers), then one block per task
    (global router restricted to the task vs that task's own router, both vs the task's best static)."""
    if picks.empty:
        return pd.DataFrame()
    parts = [headline_table(picks[picks["scope"].isin([GLOBAL, PER_TASK])], te, tr, scale, curves, "ALL")]
    if len(tasks) > 1:
        for t in tasks:
            te_t, tr_t = te.subset(te.task == t), tr.subset(tr.task == t)
            sub = picks[picks["scope"].isin([GLOBAL, t]) & picks["qid"].isin(set(te_t.qids))]
            parts.append(headline_table(sub, te_t, tr_t, scale, curves, f"task={t}"))
    h = pd.concat(parts, ignore_index=True)
    # Holm-Bonferroni within each family: all router variants vs best static, per evaluation set and mode
    h["dQ_p_holm"] = np.nan
    for _, idx in h.groupby(["eval_on", "mode"]).groups.items():
        h.loc[idx, "dQ_p_holm"] = holm(h.loc[idx, "dQ_p"].to_numpy())
    return h


def headline_table(picks: pd.DataFrame, te: Grid, tr: Grid, scale: float, curves: pd.DataFrame,
                   eval_on: str = "ALL") -> pd.DataFrame:
    """Router (seed-averaged per query) vs best static (chosen on train), paired bootstrap stratified by dataset.
    Quality is reported micro (per query), macro over datasets, and (with >1 task) macro over tasks, so that one
    task's metric scale (F1 / accuracy / EM) cannot dominate a pooled number."""
    ref_pick = static_pick(tr, len(te.qids), 0.0, scale)
    ref_q, ref_c = realise(te, ref_pick)
    oracle = curves[(curves["policy"] == "oracle_lambda") & (curves["split"] == "test") & (curves["group"] == eval_on)]
    multi_task = len(np.unique(te.task)) > 1
    rows = []
    for (pol, mode), sub in picks.groupby(["policy", "mode"]):
        per_q = sub.groupby("qid")[["quality", "cost"]].mean().loc[te.qids]
        bq = paired_bootstrap(per_q["quality"].to_numpy(), ref_q, strata=te.dataset)
        bq_ds = paired_bootstrap(per_q["quality"].to_numpy(), ref_q, strata=te.dataset, statistic="macro")
        bq_t = (paired_bootstrap(per_q["quality"].to_numpy(), ref_q, strata=te.task, statistic="macro")
                if multi_task else None)
        bc = paired_bootstrap(per_q["cost"].to_numpy(), ref_c, strata=te.dataset)
        q_or = quality_at_cost(oracle["cost"].to_numpy(), oracle["quality"].to_numpy(), ref_c.mean())
        gap = (bq.mean_a - bq.mean_b) / (q_or - bq.mean_b) if q_or - bq.mean_b > 1e-9 else float("nan")
        rows.append({"eval_on": eval_on, "scope": sub["scope"].iloc[0] if "scope" in sub else GLOBAL,
                     "policy": pol, "mode": mode, "n": len(te.qids), "n_seeds": sub["seed"].nunique(),
                     "router_quality": bq.mean_a, "static_quality": bq.mean_b, "dQ": bq.diff,
                     "dQ_ci_low": bq.ci_low, "dQ_ci_high": bq.ci_high, "dQ_p": bq.p_value,
                     "dQ_macro_dataset": bq_ds.diff, "dQ_macro_dataset_ci_low": bq_ds.ci_low,
                     "dQ_macro_dataset_ci_high": bq_ds.ci_high,
                     "dQ_macro_task": bq_t.diff if bq_t else np.nan,
                     "dQ_macro_task_ci_low": bq_t.ci_low if bq_t else np.nan,
                     "dQ_macro_task_ci_high": bq_t.ci_high if bq_t else np.nan,
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
    ap.add_argument("--no-per-task", action="store_true", help="train only the global router (skip per-task routers)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    run(cfg, args.models, args.feature_sets, args.seeds or cfg["router_seeds"], reuse=not args.no_reuse,
        per_task=not args.no_per_task)


if __name__ == "__main__":
    main()
