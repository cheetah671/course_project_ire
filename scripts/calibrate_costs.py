"""Fit the cost weights from MEASURED timings and write configs/cost_weights_measured.yaml.

Inputs
  --timings        calibration_timings.jsonl from `python -m ragroute.harness.calibrate` (generation + rerank)
  --stage-timings  stage_timings.jsonl of a run on the same hardware (AutoRAG's per-query retrieval time,
                   and a fallback rerank time if the calibration file has none)

Model (units: GPU-milliseconds, i.e. wall ms x GPUs used)
  generation, per generator g, one equation per timed batch:
      gpu_ms = a_in[g] * input_tokens + a_out[g] * output_tokens + b_g * calls        (non-negative least squares)
  b   = mean of b_g over generators (cost_weights.yaml has one b; per-generator values are in the report)
  c   = GPU-ms per query-passage pair scored by the cross-encoder
  d   = mean wall ms per index query over BM25 and dense retrieval (AutoRAG execution_time; BM25 runs on CPU,
        so d mixes CPU and GPU time; it is small next to generation, and T8 tests sensitivity to it)

Once configs/cost_weights_measured.yaml exists, load_cost_weights() uses its "measured" set as the default.
The placeholder sets stay for the sensitivity table.

Usage: python scripts/calibrate_costs.py --timings RESULTS/calibration/calibration_timings.jsonl \
           --stage-timings DATA/cache/stage_timings.jsonl [--out configs/cost_weights_measured.yaml]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def fit_generation(rows: list[dict]) -> dict:
    """Per generator: NNLS of GPU-ms on (input tokens, output tokens, calls). Returns weights and fit quality."""
    out = {}
    for g in sorted({r["generator"] for r in rows}):
        rs = [r for r in rows if r["generator"] == g]
        X = np.array([[r["input_tokens"], r["output_tokens"], r["n_calls"]] for r in rs], float)
        y = np.array([r["seconds"] * 1000.0 * r.get("gpus", 1) for r in rs], float)
        if len({(r["k"], r["max_tokens"]) for r in rs}) < 3:
            raise ValueError(f"{g}: need >= 3 distinct (k, max_tokens) cells to separate a_in, a_out and b")
        coef, _ = nnls(X, y)
        pred = X @ coef
        ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
        out[g] = {"a_in": float(coef[0]), "a_out": float(coef[1]), "b": float(coef[2]),
                  "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                  "max_rel_residual": float(np.max(np.abs(y - pred) / y)), "n_batches": len(rs),
                  "backend": rs[0].get("backend"), "gpu": rs[0].get("gpu"), "host": rs[0].get("host")}
    return out


def fit_rerank(cal_rows: list[dict], stage_rows: list[dict]) -> tuple[float, str]:
    rr = [r for r in cal_rows if r.get("component") == "rerank"]
    if rr:
        return 1000.0 * sum(r["seconds"] * r.get("gpus", 1) for r in rr) / sum(r["pairs"] for r in rr), "calibration"
    st = [r for r in stage_rows if str(r.get("stage", "")).startswith("rerank_") and r.get("pairs")]
    if st:
        return 1000.0 * sum(r["seconds"] for r in st) / sum(r["pairs"] for r in st), "stage_timings"
    raise ValueError("no rerank timings in either file")


def fit_index(stage_rows: list[dict]) -> float:
    per = {r["stage"]: r["sec_per_query_autorag"] for r in stage_rows
           if r.get("stage") in ("retrieval_bm25", "retrieval_dense") and r.get("sec_per_query_autorag") is not None}
    if len(per) < 2:
        raise ValueError(f"need retrieval_bm25 and retrieval_dense timings, found {sorted(per)}")
    return 1000.0 * float(np.mean(list(per.values())))


def fit_weights(cal_rows: list[dict], stage_rows: list[dict]) -> tuple[dict, dict]:
    gen = fit_generation([r for r in cal_rows if r.get("component") == "generate"])
    c, c_src = fit_rerank(cal_rows, stage_rows)
    weights = {"a_in": {g: round(v["a_in"], 6) for g, v in gen.items()},
               "a_out": {g: round(v["a_out"], 6) for g, v in gen.items()},
               "b": round(float(np.mean([v["b"] for v in gen.values()])), 6),
               "c": round(c, 6), "d": round(fit_index(stage_rows), 6)}
    report = {"generation_fit": gen, "rerank_source": c_src,
              "units": "GPU-ms per prompt token (a_in), per generated token (a_out), per call (b), "
                       "per rerank pair (c); wall ms per index query (d)"}
    return weights, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--stage-timings", required=True)
    ap.add_argument("--out", default=str(REPO / "configs" / "cost_weights_measured.yaml"))
    args = ap.parse_args()
    cal, stage = read_jsonl(args.timings), read_jsonl(args.stage_timings)
    if not cal:
        raise SystemExit(f"no calibration rows in {args.timings}")
    weights, report = fit_weights(cal, stage)
    doc = {"weights": weights, "info": {**report, "timings_file": str(args.timings),
                                        "stage_timings_file": str(args.stage_timings)}}
    Path(args.out).write_text("# Written by scripts/calibrate_costs.py from measured timings. Do not edit by hand.\n"
                              + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(doc, sort_keys=False))
    for g, v in report["generation_fit"].items():
        if v["r2"] < 0.95:
            print(f"WARNING {g}: generation fit R^2 = {v['r2']:.3f} < 0.95; inspect the timings before using them")


if __name__ == "__main__":
    main()
