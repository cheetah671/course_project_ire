"""Prove the staged runner reproduces AutoRAG's own end-to-end pipeline.

Runs AutoRAG's Evaluator on full/<config_id>.yaml (one module per node) for a small query subset in a
separate project dir, then compares generated answers with the staged cache for the same qids.
Run for at least: one hybrid+rerank config and one bm25 no-rerank config, for the small generator.

Expected: identical retrieved ids; generated text identical for (almost) all queries. Greedy vLLM output
can still differ for a few items because batch composition changes floating-point reduction order;
report the agreement rate in the appendix (expected >= 0.95, VERIFY).

Usage: python -m ragroute.harness.check_equivalence --config-id hybrid-k3-rr1-qwen1p5b --n 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from ragroute.config import load_experiment
from ragroute.eval.metrics import extract_answer, normalize_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    from autorag.evaluator import Evaluator

    cfg = load_experiment(args.config)
    data = Path(cfg["paths"]["data_root"])
    cache = Path(cfg["paths"]["cache_dir"])
    main_project = Path(cfg["paths"]["project_dir"])
    check_dir = data / f"equiv_{args.config_id}"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    (check_dir / "resources").mkdir(parents=True)
    for pkl in (main_project / "resources").glob("bm25_*.pkl"):  # reuse BM25 index; Chroma path is absolute
        shutil.copy2(pkl, check_dir / "resources" / pkl.name)

    meta = pd.read_parquet(data / "qa_meta.parquet", columns=["qid", "task"])
    qa = pd.read_parquet(main_project / "data" / "qa.parquet")
    qa = qa[qa["qid"].isin(meta.loc[meta["task"] == "qa", "qid"])].head(args.n)
    sub_qa = check_dir / "qa_subset.parquet"
    qa.to_parquet(sub_qa, index=False)

    ev = Evaluator(str(sub_qa), str(main_project / "data" / "corpus.parquet"), project_dir=str(check_dir))
    ev.start_trial(str(Path(cfg["paths"]["autorag_yaml_dir"]) / "full" / f"{args.config_id}.yaml"))
    trial = json.loads((check_dir / "trial.json").read_text())[-1]["trial_name"]
    gen = pd.read_parquet(check_dir / trial / "post_retrieve_node_line" / "generator" / "0.parquet")
    autorag_out = pd.DataFrame({"qid": qa["qid"].values, "autorag_text": gen["generated_texts"].values})

    # passages that reached the prompt in AutoRAG's run
    from ragroute.config import RagConfig
    from ragroute.harness.run_stages import NODE_DIR, SUFFIX

    c = RagConfig.from_id(args.config_id)
    if c.rerank:
        ids_df, col = pd.read_parquet(check_dir / trial / "post_retrieve_node_line" / "passage_reranker" / "0.parquet"), "retrieved_ids"
    else:
        ids_df = pd.read_parquet(check_dir / trial / "retrieve_node_line" / NODE_DIR[c.retriever] / "0.parquet")
        col = f"retrieved_ids{SUFFIX[c.retriever]}"
    autorag_out["autorag_ids"] = ids_df[col].map(list).values

    staged = pd.read_parquet(cache / "generation" / f"{args.config_id}.parquet")
    m = autorag_out.merge(staged[["qid", "generated_text", "retrieved_ids"]], on="qid")
    same_ids = (m["autorag_ids"].map(list) == m["retrieved_ids"].map(list)).mean()
    same_raw = (m["autorag_text"].str.strip() == m["generated_text"].str.strip()).mean()
    same_ans = (m["autorag_text"].map(lambda s: normalize_answer(extract_answer(s)))
                == m["generated_text"].map(lambda s: normalize_answer(extract_answer(s)))).mean()
    report = {"config_id": args.config_id, "n": len(m), "retrieved_ids_agreement": float(same_ids),
              "raw_text_agreement": float(same_raw),
              "normalised_answer_agreement": float(same_ans)}
    out = Path(cfg["paths"]["results_dir"]) / f"equivalence_{args.config_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
