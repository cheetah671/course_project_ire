"""Report figures (PDF + PNG into {results}/figures/). Static matplotlib figures for the written report.

Colour: three validated categorical slots (blue, orange, aqua; passes all-pairs CVD checks) for Router A,
Router B and the oracle. Baselines are neutral greys told apart by dash style AND a direct text label,
so identity never depends on colour alone (aqua is below 3:1 contrast, so every curve is direct-labelled).

F2 pareto_test          24 static configs + static frontier + baselines + routers + oracle (test split)
F3 pareto_by_dataset    the same, one panel per dataset (small multiples)
F4 oracle_winrate       share of queries where each config is oracle-best, per hops type (heatmap)
F6 ablation_gap_closed  gap closed at iso-cost per feature set x model (mean +- seed sd)
F7 importance           GBDT gain by feature group
Usage: python -m ragroute.analysis.plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ragroute.config import load_experiment  # noqa: E402
from ragroute.oracle import oracle_best, wide  # noqa: E402
from ragroute.router.policies import pareto_frontier  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"router_A": "#2a78d6", "router_B": "#eb6834", "oracle": "#1baf7a"}
BASELINE_STYLE = {"static_lambda": ("#52514e", "-"), "per_dataset_static": ("#8a8984", "--"),
                  "adaptive_rag_style": ("#8a8984", ":")}
LABELS = {"static_lambda": "Best static (frontier)", "per_dataset_static": "Per-dataset static",
          "adaptive_rag_style": "Adaptive-RAG-style", "oracle_lambda": "Oracle"}

plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "font.size": 9, "lines.linewidth": 2})


def _frontier_xy(df):
    idx = pareto_frontier(df["cost"].to_numpy(), df["quality"].to_numpy())
    return df["cost"].to_numpy()[idx], df["quality"].to_numpy()[idx]


def _direct_labels(ax, ends, min_gap_frac: float = 0.06):
    """Label each coloured curve at its right end; nudge labels apart vertically so they never overlap."""
    if not ends:
        return
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap = (y1 - y0) * min_gap_frac
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text, color in ends:
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.plot([x], [y], "o", ms=4, color=color, zorder=4)
        ax.annotate(text, (x, y), xytext=(x + 0.015 * (x1 - x0), ty), textcoords="data", va="center", ha="left", fontsize=8, color=INK,
                    xycoords="data", annotation_clip=False,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))


def pareto_panel(ax, curves: pd.DataFrame, group: str, router_a: str, router_b: str, labels: bool = True):
    c = curves[(curves["split"] == "test") & (curves["group"] == group)]
    st = c[c["policy"] == "static_config"]
    ax.scatter(st["cost"], st["quality"], s=22, color="#b9b8b2", edgecolor=SURFACE, linewidth=1, zorder=2,
               label="Static configs (24)")
    for pol, (col, ls) in BASELINE_STYLE.items():
        d = c[c["policy"] == pol]
        if len(d):
            x, y = _frontier_xy(d)
            ax.step(x, y, where="post", color=col, ls=ls, lw=1.5, label=LABELS[pol])
    ends = []
    for pol, key, name in ((router_a, "router_A", "Router A"), (router_b, "router_B", "Router B")):
        d = c[c["policy"] == pol]
        if len(d):
            d = d.groupby("lambda")[["cost", "quality"]].mean().reset_index()  # mean over seeds
            x, y = _frontier_xy(d)
            ax.step(x, y, where="post", color=SERIES[key], label=f"{name} ({'question' if key == 'router_A' else '+retrieval'})")
            ends.append((x[-1], y[-1], name, SERIES[key]))
    d = c[c["policy"] == "oracle_lambda"]
    if len(d):
        x, y = _frontier_xy(d)
        ax.step(x, y, where="post", color=SERIES["oracle"], label="Oracle (upper bound)")
        ax.plot(x, y, "o", ms=4, color=SERIES["oracle"], zorder=4)  # visible even if the frontier is one point
        ends.append((x[-1], y[-1], "Oracle", SERIES["oracle"]))
    if labels:
        _direct_labels(ax, ends)
    ax.set_xlabel("Mean marginal cost per query (cost units)")
    ax.set_ylabel("Mean quality (F1 / accuracy)")
    ax.set_title(group if group != "ALL" else "All datasets (test)", color=INK, fontsize=10, loc="left")


def fig_pareto(curves, out: Path, router_a: str, router_b: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    pareto_panel(ax, curves, "ALL", router_a, router_b)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    _save(fig, out / "F2_pareto_test")
    groups = sorted(g for g in curves["group"].unique() if g.startswith("dataset="))
    if groups:
        fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups) + 2.2, 3.4), squeeze=False, sharey=True)
        for i, (ax, g) in enumerate(zip(axes[0], groups)):
            pareto_panel(ax, curves, g, router_a, router_b, labels=False)
            if i:
                ax.set_ylabel("")
        axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        _save(fig, out / "F3_pareto_by_dataset")


def fig_oracle_winrate(runs: pd.DataFrame, out: Path):
    sub = runs[runs["split"] == "test"]
    rows = []
    for hops, g in sub.groupby("hops"):
        Q, C = wide(g, "quality"), wide(g, "cost")
        pick = oracle_best(Q.to_numpy(), C.to_numpy())
        share = np.bincount(pick, minlength=Q.shape[1]) / len(pick)
        rows.append(pd.Series(share, index=Q.columns, name=hops))
    M = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.4 * len(M)))
    im = ax.imshow(M.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=75, fontsize=7)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of queries (oracle-best, ties -> cheapest)")
    ax.set_title("Which config is oracle-best, by question type (test)", loc="left", fontsize=10, color=INK)
    _save(fig, out / "F4_oracle_winrate")


def fig_ablation(headline: pd.DataFrame, out: Path):
    h = headline[headline["mode"] == "iso_cost"].copy()
    h[["fs", "model"]] = h["policy"].str.replace("router_", "", regex=False).str.rsplit("_", n=1, expand=True)
    piv = h.pivot(index="fs", columns="model", values="gap_closed")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.8 / max(len(piv.columns), 1)
    for i, m in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * width, piv[m], width=width - 0.02, color=list(SERIES.values())[i % 3], label=m)
    ax.set_xticks(np.arange(len(piv)) + width * (len(piv.columns) - 1) / 2, piv.index)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Gap closed at iso-cost\n(router - static) / (oracle - static)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "F6_ablation_gap_closed")


def fig_importance(imp: dict, out: Path):
    if not imp:
        return
    df = pd.DataFrame(imp).T.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.35 * len(df)))
    left = np.zeros(len(df))
    for i, col in enumerate(df.columns):
        ax.barh(df.index, df[col], left=left, color=["#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#8a8984"][i % 5],
                edgecolor=SURFACE, linewidth=2, label=col)
        left += df[col].to_numpy()
    ax.set_xlabel("share of GBDT gain")
    ax.legend(frameon=False, fontsize=8, ncol=len(df.columns), loc="upper center", bbox_to_anchor=(0.5, -0.35))
    _save(fig, out / "F7_importance")


def _save(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--router-a", default="router_A_gbdt")
    ap.add_argument("--router-b", default="router_B_gbdt")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    res = Path(cfg["paths"]["results_dir"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    curves = pd.read_parquet(res / "router" / "curves.parquet")
    fig_pareto(curves, out, args.router_a, args.router_b)
    fig_oracle_winrate(pd.read_parquet(res / "runs_scored.parquet"), out)
    fig_ablation(pd.read_csv(res / "router" / "headline.csv"), out)
    fig_importance(json.loads((res / "router" / "importance.json").read_text()), out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
