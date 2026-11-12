"""Report tables (written as CSV + LaTeX into {results}/tables/). Numbers come only from result files.

T1 datasets_corpus      per dataset: task, metric, sampled n, train/dev/test, gold-mapping rate; corpus composition
T2 static_grid          per config x dataset: EM, F1 (95% bootstrap CI), cost components, scalar cost
T3 oracle_labels        oracle ceiling vs best static; label statistics (all-fail / all-succeed / mixed)
T6 breakdown            router vs best static per dataset and per hops type (paired bootstrap)
T7 best_by_qtype        best static config per question type with stability + margin CI
T8 weight_sensitivity   static cost ranking stability (Kendall tau) and frontier membership under weight draws
T9 offline_vs_marginal  offline cost, marginal cost, amortised total under several amortisation factors
T10 latency             p50/p95 per config (same session only)
T3c / T4c chance-corrected FEVER: FEVER accuracy has a 0.5 chance floor while QA F1 / zsRE EM start near 0, so a
                        pooled mean mixes scales. Re-report with FEVER as 2*acc - 1 (0 = chance), routing unchanged.
T8b router weight sensitivity: router iso-cost dQ and iso-quality saving under every alternative weight set
                        (router/train_eval.py --weight-set <name>, output dirs router_w-<name>/)
Router headline (T4) and ablations (T5) are produced by router/train_eval.py and summarised here.

Usage: python -m ragroute.analysis.tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ragroute.config import load_cost_weights, load_experiment
from ragroute.cost import amortised_offline_per_query, random_weight_draws, scalar_cost
from ragroute.oracle import label_statistics, oracle_best, wide
from ragroute.router.policies import pareto_frontier
from ragroute.stats.bootstrap import best_config_stability, bootstrap_mean_ci, holm, paired_bootstrap


def static_grid(runs: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    sub = runs[runs["split"] == split]
    rows = []
    for (ds, cid), g in sub.groupby(["dataset", "config_id"]):
        em, f1 = bootstrap_mean_ci(g["em"]), bootstrap_mean_ci(g["quality"])
        rows.append({"dataset": ds, "config_id": cid, "EM": em["mean"], "quality": f1["mean"],
                     "quality_ci": f"[{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]",
                     "in_tok": g["input_tokens"].mean(), "out_tok": g["output_tokens"].mean(),
                     "rerank_pairs": g["rerank_passes"].mean(), "index_q": g["index_queries"].mean(),
                     "cost": g["cost"].mean(), "cost_cv": g["cost"].std() / g["cost"].mean()})
    return pd.DataFrame(rows)


def oracle_vs_static(runs: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    train = runs[runs["split"] == "train"]
    best_static = train.groupby("config_id")["quality"].mean().idxmax()
    sub = runs[runs["split"] == split]
    rows = []
    for ds, g in list(sub.groupby("dataset")) + [("ALL", sub)]:
        Q, C = wide(g, "quality"), wide(g, "cost")
        pick = oracle_best(Q.to_numpy(), C.to_numpy())
        oq = Q.to_numpy()[np.arange(len(pick)), pick]
        sq = Q[best_static].to_numpy()
        b = paired_bootstrap(oq, sq, strata=None if ds != "ALL" else g.drop_duplicates("qid").set_index("qid").loc[Q.index, "dataset"].to_numpy())
        rows.append({"dataset": ds, "best_static(train)": best_static, "static_quality": b.mean_b,
                     "oracle_quality": b.mean_a, "headroom": b.diff, "ci": f"[{b.ci_low:.3f}, {b.ci_high:.3f}]"})
    return pd.DataFrame(rows).merge(label_statistics(sub), on="dataset", how="left")


def breakdown(picks: pd.DataFrame, runs: pd.DataFrame, policy: str, mode: str) -> pd.DataFrame:
    test = runs[runs["split"] == "test"]
    best_static = runs[runs["split"] == "train"].groupby("config_id")["quality"].mean().idxmax()
    ref = test[test["config_id"] == best_static].set_index("qid")
    sel = picks[(picks["policy"] == policy) & (picks["mode"] == mode)].groupby("qid")["quality"].mean()
    cols = ["dataset", "hops", "qtype"] + (["task"] if "task" in ref.columns else [])
    df = ref[cols + ["quality"]].join(sel.rename("router"), how="inner")
    rows = []
    for col in cols:
        if col == "task" and df["task"].nunique() < 2:
            continue
        for val, g in df.groupby(col):
            b = paired_bootstrap(g["router"].to_numpy(), g["quality"].to_numpy())
            rows.append({"by": col, "value": val, "n": len(g), "router": b.mean_a, "static": b.mean_b,
                         "dQ": b.diff, "ci_low": b.ci_low, "ci_high": b.ci_high, "p": b.p_value})
    out = pd.DataFrame(rows)
    if len(out):  # Holm-Bonferroni within each breakdown family (all datasets; all hops types; ...)
        out["p_holm"] = out.groupby("by")["p"].transform(lambda p: holm(p.to_numpy()))
    return out


def best_by_qtype(runs: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    sub = runs[runs["split"] == split]
    rows = []
    for (ds, qt), g in sub.groupby(["dataset", "qtype"]):
        Q = wide(g, "quality")
        if len(Q) < 30:
            continue
        r = best_config_stability(Q.to_numpy(), list(Q.columns))
        rows.append({"dataset": ds, "qtype": qt, "n": len(Q), "best": r["best"], "runner_up": r["runner_up"],
                     "best_win_freq": r["win_freq"][r["best"]], "margin": r["margin"]["diff"],
                     "margin_ci": f"[{r['margin']['ci_low']:.3f}, {r['margin']['ci_high']:.3f}]",
                     "claim_supported": r["claim_supported"]})
    return pd.DataFrame(rows)


def weight_sensitivity(runs: pd.DataFrame, weights: dict, split: str = "test") -> pd.DataFrame:
    sub = runs[runs["split"] == split]
    q = sub.groupby("config_id")["quality"].mean()
    base_w = weights["weight_sets"][weights["default"]]
    base = sub.assign(c=scalar_cost(sub, base_w)).groupby("config_id")["c"].mean().loc[q.index]
    base_front = set(q.index[pareto_frontier(base.to_numpy(), q.to_numpy())])
    s = weights["sensitivity"]
    rows = []
    named = [(n, w) for n, w in weights["weight_sets"].items()]
    draws = [(f"draw{i}", w) for i, w in enumerate(random_weight_draws(base_w, s["n_random_draws"], s["log_uniform_factor"], 0))]
    for name, w in named + draws:
        cost = sub.assign(c=scalar_cost(sub, w)).groupby("config_id")["c"].mean().loc[q.index]
        front = set(q.index[pareto_frontier(cost.to_numpy(), q.to_numpy())])
        rows.append({"weights": name, "kendall_tau_cost_rank": kendalltau(base, cost).statistic,
                     "frontier_jaccard": len(front & base_front) / len(front | base_front),
                     "frontier": ";".join(sorted(front))})
    return pd.DataFrame(rows)


def offline_vs_marginal(runs: pd.DataFrame, offline: dict, factors) -> pd.DataFrame:
    marg = runs[runs["split"] == "test"].groupby("config_id")["cost"].mean()
    rows = []
    for f in factors:
        amort = amortised_offline_per_query(offline, f)
        for cid, m in marg.items():
            # dense/hybrid configs need the vector index; bm25-only needs only the BM25 index
            needs_dense = not cid.startswith("bm25")
            a = amort if needs_dense else amortised_offline_per_query(
                {k: v for k, v in offline.items() if not k.startswith("embed")}, f)
            rows.append({"queries_per_build": f, "config_id": cid, "marginal": m, "amortised_offline": a, "total": m + a})
    return pd.DataFrame(rows)


def latency_summary(lat: pd.DataFrame) -> pd.DataFrame:
    """T10; refuses rows from more than one session. Same table as `ragroute.eval.latency summarise`."""
    from ragroute.eval.latency import summarise_rows

    return summarise_rows(lat)


METRIC = {"qa": "token F1", "verification": "accuracy", "slot_filling": "exact match"}


def datasets_corpus(data: Path, res: Path) -> pd.DataFrame:
    """T1: one row per dataset plus corpus rows (sub-corpus composition) from corpus_stats.json."""
    meta = pd.read_parquet(data / "qa_meta.parquet")
    splits = pd.read_parquet(data / "splits.parquet")[["qid", "split"]]
    meta = meta.merge(splits, on="qid", validate="1:1")
    rows = []
    for ds, g in meta.groupby("dataset"):
        n_split = g["split"].value_counts()
        rows.append({"item": ds, "task": g["task"].iloc[0], "metric": METRIC.get(g["task"].iloc[0], ""),
                     "n": len(g), "train": int(n_split.get("train", 0)), "dev": int(n_split.get("dev", 0)),
                     "test": int(n_split.get("test", 0)),
                     "gold_mapped_pct": round(100 * float(g["has_gold"].mean()), 1) if "has_gold" in g else np.nan,
                     "qtypes": "; ".join(f"{k}={v}" for k, v in g["qtype"].value_counts().items())})
    stats_path = data / "corpus_stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        for k in ("full_corpus_passages", "subcorpus_passages", "n_mined", "n_gold_neighbours", "n_random_fill",
                  "gold_titles_needed", "gold_titles_found"):
            if k in st:
                rows.append({"item": f"corpus: {k}", "n": st[k]})
    leak = res / "leakage_report.json"
    if leak.exists():
        for k, v in json.loads(leak.read_text()).items():
            if isinstance(v, (int, float)):
                rows.append({"item": f"leakage: {k}", "n": v})
    return pd.DataFrame(rows)


def chance_corrected(q: pd.Series, task: pd.Series) -> pd.Series:
    """FEVER accuracy -> 2*acc - 1 (0 = chance on 2 balanced labels); other tasks unchanged."""
    return q.where(task != "verification", 2 * q - 1)


def fever_chance_corrected_static(runs: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """T3c: per task, best static (chosen on train, raw metric) and oracle-best, raw and chance-corrected,
    plus the macro-over-task mean both ways."""
    train = runs[runs["split"] == "train"]
    best_static = train.groupby("config_id")["quality"].mean().idxmax()
    sub = runs[runs["split"] == split]
    rows = []
    for task, g in sub.groupby("task"):
        Q, C = wide(g, "quality"), wide(g, "cost")
        pick = oracle_best(Q.to_numpy(), C.to_numpy())
        oq = pd.Series(Q.to_numpy()[np.arange(len(pick)), pick])
        sq = Q[best_static].reset_index(drop=True)
        t = pd.Series(task, index=oq.index)
        rows.append({"task": task, "n": len(Q), "best_static": best_static,
                     "static_raw": sq.mean(), "oracle_raw": oq.mean(),
                     "static_cc": chance_corrected(sq, t).mean(), "oracle_cc": chance_corrected(oq, t).mean()})
    df = pd.DataFrame(rows)
    macro = {"task": "MACRO over tasks", "n": int(df["n"].sum()), "best_static": best_static}
    for c in ("static_raw", "oracle_raw", "static_cc", "oracle_cc"):
        macro[c] = df[c].mean()
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    df["headroom_raw"] = df["oracle_raw"] - df["static_raw"]
    df["headroom_cc"] = df["oracle_cc"] - df["static_cc"]
    return df


def fever_chance_corrected_routers(picks: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """T4c: every router on ALL test queries vs the best static config (chosen on train), macro over tasks,
    with FEVER raw and chance-corrected. The routing decisions are the main run's; only the scoring changes."""
    test = runs[runs["split"] == "test"]
    if test["task"].nunique() < 2 or picks.empty:
        return pd.DataFrame()
    best_static = runs[runs["split"] == "train"].groupby("config_id")["quality"].mean().idxmax()
    ref = test[test["config_id"] == best_static].set_index("qid")[["quality", "task"]]
    rows = []
    for (pol, mode), sub in picks[picks["scope"].isin(["global", "per_task"])].groupby(["policy", "mode"]):
        per_q = sub.groupby("qid")["quality"].mean().reindex(ref.index)
        for name, f in (("raw", lambda q: q), ("chance_corrected", lambda q: chance_corrected(q, ref["task"]))):
            b = paired_bootstrap(f(per_q).to_numpy(), f(ref["quality"]).to_numpy(), strata=ref["task"].to_numpy(),
                                 statistic="macro")
            rows.append({"policy": pol, "mode": mode, "fever_scoring": name, "router_macro_task": b.mean_a,
                         "static_macro_task": b.mean_b, "dQ_macro_task": b.diff, "ci_low": b.ci_low,
                         "ci_high": b.ci_high, "p": b.p_value})
    return pd.DataFrame(rows)


def router_weight_sensitivity(res: Path, policies=("router_A_gbdt", "router_B_gbdt")) -> pd.DataFrame:
    """T8b: the headline (ALL test queries, global scope) for each cost-weight set the routers were re-run with."""
    rows = []
    for d in [res / "router", *sorted(res.glob("router_w-*"))]:
        h = d / "headline.csv"
        if not h.exists():
            continue
        df = pd.read_csv(h)
        if "eval_on" in df:
            df = df[(df["eval_on"] == "ALL") & (df["scope"] == "global")]
        df = df[df["policy"].isin(policies) & df["mode"].isin(["iso_cost", "iso_quality"])]
        ws = d.name.split("router_w-")[1] if "router_w-" in d.name else "default"
        for r in df.itertuples():
            rows.append({"weight_set": ws, "policy": r.policy, "mode": r.mode, "dQ": r.dQ,
                         "dQ_ci": f"[{r.dQ_ci_low:.3f}, {r.dQ_ci_high:.3f}]", "cost_saving_pct": r.cost_saving_pct,
                         "gap_closed": r.gap_closed})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default="router_B_gbdt")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    res = Path(cfg["paths"]["results_dir"])
    out = res / "tables"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(res / "runs_scored.parquet")
    weights = load_cost_weights()

    def save(df, name):
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.tex").write_text(df.to_latex(index=False, float_format="%.3f"))
        print(f"== {name}\n{df.head(30).to_string()}\n")

    save(datasets_corpus(Path(cfg["paths"]["data_root"]), res), "T1_datasets_corpus")
    save(static_grid(runs), "T2_static_grid")
    save(oracle_vs_static(runs), "T3_oracle_vs_static")
    if runs["task"].nunique() > 1:
        save(fever_chance_corrected_static(runs), "T3c_fever_chance_corrected_static")
    save(best_by_qtype(runs), "T7_best_by_qtype")
    ws = weight_sensitivity(runs, weights)
    save(ws, "T8_weight_sensitivity_draws")
    save(ws.drop(columns="frontier").describe().reset_index(), "T8_weight_sensitivity_summary")
    picks_path = res / "router" / "picks.parquet"
    if picks_path.exists():
        picks = pd.read_parquet(picks_path)
        for mode in ("iso_cost", "iso_quality"):
            save(breakdown(picks, runs, args.policy, mode), f"T6_breakdown_{args.policy}_{mode}")
        head = pd.read_csv(res / "router" / "headline.csv")
        save(head, "T4_T5_router_headline_and_ablations")
        cc = fever_chance_corrected_routers(picks, runs)
        if len(cc):
            save(cc, "T4c_fever_chance_corrected_routers")
        rws = router_weight_sensitivity(res)
        if len(rws):
            save(rws, "T8b_router_weight_sensitivity")
    offline_path = res / "offline_cost.json"
    if offline_path.exists():
        save(offline_vs_marginal(runs, json.loads(offline_path.read_text()),
                                 weights["amortisation"]["queries_per_build"]), "T9_offline_vs_marginal")
    lat_path = res / "latency.parquet"
    if lat_path.exists():
        save(latency_summary(pd.read_parquet(lat_path)), "T10_latency")


if __name__ == "__main__":
    main()
