"""Compare two generation caches config by config: e.g. vLLM vs the pilot's llama.cpp outputs, or a greedy rerun.

For every config present in both {cache}/<a>/ and {cache}/<b>/: identical-text rate, mean quality (the router
target, from score_row) on each side, input-token agreement (same prompts?), and wall-clock from stage_timings.
Usage: python -m ragroute.harness.compare_generations --a generation --b generation_vllm --out results/x.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ragroute.config import load_experiment
from ragroute.eval.metrics import score_row


def _timings(cache: Path) -> list[dict]:
    path = cache / "stage_timings.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _timing_for(records: list[dict], config_id: str, subdir: str) -> dict | None:
    """Latest record for this config written into `subdir`. Records from before the `subdir` field existed are
    taken to be the main 'generation' run, and the earliest one is used (later ones were reruns elsewhere)."""
    stage = f"generate_{config_id}"
    tagged = [r for r in records if r.get("stage") == stage and r.get("subdir") == subdir]
    if tagged:
        return tagged[-1]
    untagged = [r for r in records if r.get("stage") == stage and "subdir" not in r]
    return untagged[0] if untagged and subdir == "generation" else None


def compare(cfg: dict, sub_a: str, sub_b: str) -> list[dict]:
    data, cache = Path(cfg["paths"]["data_root"]), Path(cfg["paths"]["cache_dir"])
    meta = pd.read_parquet(data / "qa_meta.parquet", columns=["qid", "task", "answers"]).set_index("qid")
    timings = _timings(cache)
    rows = []
    for path_b in sorted((cache / sub_b).glob("*.parquet")):
        path_a = cache / sub_a / path_b.name
        if not path_a.exists():
            continue
        a = pd.read_parquet(path_a).set_index("qid")
        b = pd.read_parquet(path_b).set_index("qid")
        q = b.index.intersection(a.index)

        def quality(df: pd.DataFrame) -> float:
            return float(pd.Series([score_row(meta.at[i, "task"], df.at[i, "generated_text"],
                                              list(meta.at[i, "answers"]))["quality"] for i in q]).mean())

        row = {"config_id": path_b.stem, "n": int(len(q)),
               "same_text_rate": float((a.loc[q, "generated_text"] == b.loc[q, "generated_text"]).mean()),
               f"quality_{sub_a}": quality(a), f"quality_{sub_b}": quality(b),
               "input_tokens_equal_rate": float((a.loc[q, "input_tokens"] == b.loc[q, "input_tokens"]).mean()),
               f"output_tokens_mean_{sub_a}": float(a.loc[q, "output_tokens"].mean()),
               f"output_tokens_mean_{sub_b}": float(b.loc[q, "output_tokens"].mean())}
        for sub in (sub_a, sub_b):
            t = _timing_for(timings, path_b.stem, sub)
            if t:
                row[f"seconds_{sub}"] = t["seconds"]
                row[f"gpu_{sub}"] = t.get("gpu")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--a", default="generation", help="reference cache sub-folder")
    ap.add_argument("--b", required=True, help="cache sub-folder to compare against --a")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()
    rows = compare(load_experiment(args.config), args.a, args.b)
    print(pd.DataFrame(rows).to_string(index=False))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
