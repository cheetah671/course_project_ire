"""Step 1: load each enabled dataset, normalise to one schema, sample ~n queries per dataset.

Output: {data_root}/qa_raw.parquet with columns
  qid, dataset, task, hops, qtype, query, answers (list[str]), gold_titles (list[str])

Sampling procedure (reproducible):
  * drop items with no gold answer and exact duplicate questions (after normalisation) within a dataset
  * rng = numpy default_rng(seed + crc32(dataset_name))
  * if `stratify_by` is set: proportional allocation with largest-remainder rounding, then sample
    without replacement inside each stratum; otherwise simple random sampling without replacement
  * the sample is sorted by qid so the file is byte-identical across runs

Field names follow the Hugging Face dataset cards as of writing. VERIFY each loader on first run
(the script prints one normalised example per dataset).

Usage: python -m ragroute.data.prepare_qa [--config configs/experiment.yaml] [--fever-jsonl PATH]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import enabled_datasets, load_experiment
from ragroute.eval.metrics import normalize_answer


def _load_hf(spec: dict):
    from datasets import load_dataset  # imported lazily: not needed for tests

    return load_dataset(spec["hf_id"], spec.get("hf_config"), split=spec["hf_split"])


def load_nq(spec):
    ds = _load_hf(spec)
    for i, ex in enumerate(ds):
        yield {"orig_id": str(i), "query": ex["question"], "answers": list(ex["answer"]),
               "qtype": "single", "gold_titles": []}


def load_hotpotqa(spec):
    ds = _load_hf(spec)
    for ex in ds:
        titles = sorted(set(ex["supporting_facts"]["title"]))
        yield {"orig_id": ex["id"], "query": ex["question"], "answers": [ex["answer"]],
               "qtype": ex["type"], "gold_titles": titles}


def load_triviaqa(spec):
    ds = _load_hf(spec)
    for ex in ds:
        ans = ex["answer"]
        golds = list(dict.fromkeys([ans["value"], *ans.get("aliases", [])]))
        yield {"orig_id": ex["question_id"], "query": ex["question"], "answers": golds,
               "qtype": "single", "gold_titles": []}


def _as_list(x) -> list:
    """None -> []; list / tuple / numpy array (pandas-parquet rows) -> list."""
    return [] if x is None else list(x)


def load_fever(spec, jsonl_path: str | None = None):
    """KILT FEVER: claim in `input`, label in output[*].answer (SUPPORTS/REFUTES), provenance titles."""
    if jsonl_path:  # fallback: official KILT jsonl (e.g. fever-dev-kilt.jsonl; VERIFY URL)
        rows = (json.loads(line) for line in open(jsonl_path, encoding="utf-8"))
    else:
        rows = _load_hf(spec)
    for ex in rows:
        outputs = [o for o in ex["output"] if o.get("answer")]
        if not outputs:
            continue
        label = outputs[0]["answer"]
        titles = sorted({p["title"] for o in ex["output"] for p in _as_list(o.get("provenance")) if p.get("title")})
        yield {"orig_id": str(ex["id"]), "query": ex["input"], "answers": [label],
               "qtype": label.lower(), "gold_titles": titles}


LOADERS = {"nq": load_nq, "hotpotqa": load_hotpotqa, "triviaqa": load_triviaqa, "fever": load_fever}


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator, by: str | None) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    if not by:
        return df.iloc[rng.choice(len(df), size=n, replace=False)]
    counts = df[by].value_counts().sort_index()
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    remainder = n - alloc.sum()
    # largest remainder; ties broken by stratum name order (deterministic)
    for key in (exact - alloc).sort_values(ascending=False, kind="mergesort").index[:remainder]:
        alloc[key] += 1
    parts = []
    for key, k in alloc.items():
        sub = df[df[by] == key]
        parts.append(sub.iloc[rng.choice(len(sub), size=int(k), replace=False)])
    return pd.concat(parts)


def build(cfg: dict, fever_jsonl: str | None = None) -> pd.DataFrame:
    frames = []
    for name in enabled_datasets(cfg):
        spec = cfg["datasets"][name]
        loader = LOADERS[name]
        rows = list(loader(spec, fever_jsonl) if name == "fever" else loader(spec))
        df = pd.DataFrame(rows)
        n_loaded = len(df)
        df = df[df["answers"].map(lambda a: any(normalize_answer(x) for x in a))]
        df["_norm_q"] = df["query"].map(normalize_answer)
        df = df.drop_duplicates("_norm_q").drop(columns="_norm_q")
        rng = np.random.default_rng(cfg["seed"] + zlib.crc32(name.encode()))
        sample = stratified_sample(df.reset_index(drop=True), spec["n"], rng, spec.get("stratify_by"))
        sample = sample.assign(
            qid=lambda d: name + ":" + d["orig_id"].astype(str),
            dataset=name, task=spec["task"], hops=spec["hops"],
        )
        print(f"[{name}] loaded={n_loaded} after_filter={len(df)} sampled={len(sample)}")
        print(f"[{name}] example: {sample.iloc[0][['query', 'answers', 'qtype', 'gold_titles']].to_dict()}")
        frames.append(sample)
    out = pd.concat(frames).sort_values("qid").reset_index(drop=True)
    return out[["qid", "dataset", "task", "hops", "qtype", "query", "answers", "gold_titles", "orig_id"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fever-jsonl", default=None)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    out = build(cfg, args.fever_jsonl)
    path = Path(cfg["paths"]["data_root"]) / "qa_raw.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {len(out)} queries -> {path}")
    print(out.groupby(["dataset", "qtype"]).size().to_string())


if __name__ == "__main__":
    main()
