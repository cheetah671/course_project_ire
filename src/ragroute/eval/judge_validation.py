"""150-item judge validation: sample items for annotation, then score judge-vs-human agreement.

  sample : stratified sample from the DEV split (test stays untouched by humans) -> annotation CSV + parquet
           strata: dataset x (EM correct / incorrect) x (k=3 / k=10), so both easy and hard cases appear
  score  : reads 2 annotators' CSVs + judge scores (ragas_eval.py --items ...) and reports
           inter-annotator Cohen's kappa, judge-vs-human Spearman/Pearson, judge-vs-human Cohen's kappa
           (judge binarised at 0.5; human binary = label 2 'fully supported'), all with 95% bootstrap CIs.
           Also EM/F1 vs human correctness, which quantifies EM strictness (false negatives).

Decision rule (pre-registered): run RAGAS at scale only if judge-vs-human faithfulness kappa >= 0.6 (VERIFY
threshold with the team) and the 95% CI lower bound of Spearman > 0.4. Otherwise RAGAS is reported only as
an unreliable diagnostic.

Usage: python -m ragroute.eval.judge_validation sample
       python -m ragroute.eval.judge_validation score --ann annotation/ann_A.csv annotation/ann_B.csv \
              --judge results/judge_validation_ragas.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from ragroute.config import REPO_ROOT, load_experiment

N_ITEMS = 150
ANN_COLS = ["faithfulness_label", "relevance_label", "correct_label", "notes", "annotator_id",
            "adjudicated_faithfulness"]


def sample(cfg: dict) -> pd.DataFrame:
    from ragroute.eval.ragas_eval import build_items  # reuse context assembly

    res = Path(cfg["paths"]["results_dir"])
    runs = pd.read_parquet(res / "runs_scored.parquet")
    dev = runs[runs["split"] == "dev"].assign(k=lambda d: d["config_id"].str.extract(r"-k(\d+)-")[0])
    dev["stratum"] = dev["dataset"] + "|em" + dev["em"].astype(int).astype(str) + "|k" + dev["k"]
    rng = np.random.default_rng(cfg["seed"] + 150)
    strata = sorted(dev["stratum"].unique())
    per = int(np.ceil(N_ITEMS / len(strata)))
    picked = []
    for s in strata:
        g = dev[dev["stratum"] == s].drop_duplicates("qid")  # one config per query within a stratum
        picked.append(g.iloc[rng.choice(len(g), size=min(per, len(g)), replace=False)])
    items = pd.concat(picked).sample(frac=1, random_state=int(rng.integers(1 << 31))).head(N_ITEMS)
    # attach query, gold, contexts
    qa = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "qa.parquet")[["qid", "query", "generation_gt"]]
    items = items.merge(qa, on="qid")
    cache = Path(cfg["paths"]["cache_dir"]) / "generation"
    ids = pd.concat([pd.read_parquet(cache / f"{c}.parquet", columns=["qid", "config_id", "retrieved_ids"])
                     for c in items["config_id"].unique()])
    items = items.merge(ids, on=["qid", "config_id"])
    corpus = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "corpus.parquet", columns=["doc_id", "contents"])
    lookup = dict(zip(corpus["doc_id"], corpus["contents"]))
    items["contexts"] = items["retrieved_ids"].map(lambda x: [lookup[i] for i in x])
    items.insert(0, "item_id", [f"JV{i:03d}" for i in range(len(items))])
    return items


def to_annotation_csv(items: pd.DataFrame, path: Path) -> None:
    sheet = pd.DataFrame({
        "item_id": items["item_id"], "dataset": items["dataset"], "question_or_claim": items["query"],
        "gold_answers": items["generation_gt"].map(lambda g: " | ".join(g)),
        "passages": items["contexts"].map(lambda cs: "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(cs))),
        "system_answer": items["pred"],
    })
    for c in ANN_COLS:
        sheet[c] = ""
    sheet.to_csv(path, index=False)


def _boot(fn, a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        try:
            vals.append(fn(a[i], b[i]))
        except ValueError:
            continue
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(fn(a, b)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def score(ann_paths: list[str], judge_path: str, items_path: str) -> dict:
    anns = [pd.read_csv(p) for p in ann_paths]
    items = pd.read_parquet(items_path)
    judge = pd.read_parquet(judge_path)
    rep = {}
    if len(anns) >= 2:
        a, b = anns[0].set_index("item_id"), anns[1].set_index("item_id").loc[anns[0]["item_id"]]
        for col in ("faithfulness_label", "relevance_label", "correct_label"):
            rep[f"iaa_kappa_{col}"] = _boot(lambda x, y: cohen_kappa_score(x, y), a[col].astype(int), b[col].astype(int))
            rep[f"iaa_weighted_kappa_{col}"] = float(cohen_kappa_score(a[col].astype(int), b[col].astype(int), weights="quadratic"))
    # adjudicated labels: if annotators disagree the adjudication column (filled by a third person) wins
    gold = anns[0].set_index("item_id")
    if "adjudicated_faithfulness" in gold.columns:
        gold["faithfulness_label"] = gold["adjudicated_faithfulness"].fillna(gold["faithfulness_label"])
    m = items[["item_id", "qid", "config_id", "em", "f1"]].merge(judge, on=["qid", "config_id"]).set_index("item_id")
    m = m.join(gold[["faithfulness_label", "relevance_label", "correct_label"]], how="inner").dropna(subset=["faithfulness"])
    h_faith = m["faithfulness_label"].astype(float)
    rep["n_items_scored"] = int(len(m))
    rep["spearman_faithfulness"] = _boot(lambda x, y: spearmanr(x, y).statistic, m["faithfulness"], h_faith)
    rep["pearson_faithfulness"] = _boot(lambda x, y: pearsonr(x, y).statistic, m["faithfulness"], h_faith)
    rep["kappa_faithfulness_binary"] = _boot(lambda x, y: cohen_kappa_score((x >= 0.5).astype(int), (y == 2).astype(int)),
                                             m["faithfulness"], h_faith)
    if "answer_relevancy" in m:
        rep["spearman_relevancy"] = _boot(lambda x, y: spearmanr(x, y).statistic, m["answer_relevancy"], m["relevance_label"].astype(float))
    hc = m["correct_label"].astype(int)
    rep["em_vs_human_correct_kappa"] = float(cohen_kappa_score(m["em"].astype(int), hc))
    rep["em_false_negative_rate"] = float(((m["em"] == 0) & (hc == 1)).sum() / max((hc == 1).sum(), 1))
    rep["f1_vs_human_correct_spearman"] = float(spearmanr(m["f1"], hc).statistic)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--ann", nargs="*", default=[])
    ap.add_argument("--judge", default=None)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    res = Path(cfg["paths"]["results_dir"])
    items_path = res / "judge_validation_items.parquet"
    if args.cmd == "sample":
        items = sample(cfg)
        items.to_parquet(items_path, index=False)
        to_annotation_csv(items, REPO_ROOT / "annotation" / "judge_validation_to_annotate.csv")
        print(f"{len(items)} items; strata:\n{items.groupby(['dataset', 'em']).size()}")
    else:
        rep = score(args.ann, args.judge, str(items_path))
        (res / "judge_validation_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
