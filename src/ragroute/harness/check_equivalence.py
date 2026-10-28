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
    from ragroute.harness import autorag_patches

    autorag_patches.apply()  # batched Chroma id checks / inserts (large corpora)

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
    # skip_validation: AutoRAG's Validator runs a mini-trial on a sampled corpus in a temp dir but reuses our
    # absolute Chroma path; once the full index exists it returns ids missing from the sample and crashes.
    # Data is validated by build_corpus.py and by Evaluator.__init__ (schema + retrieval_gt ids) instead.
    yaml_path = str(Path(cfg["paths"]["autorag_yaml_dir"]) / "full" / f"{args.config_id}.yaml")
    if cfg["generation"]["backend"] == "llamacpp":
        # the YAML's openai_llm node talks to a llama.cpp server for this config's generator
        from ragroute.config import RagConfig as _RC
        from ragroute.harness.llamacpp import LlamaServer, ensure_gguf, patch_openai_llm

        patch_openai_llm()  # same get_result as the staged runner's LlamaCppGenerator (stock one breaks on llama.cpp)
        g = cfg["models"]["generators"][_RC.from_id(args.config_id).generator]
        lc = cfg["generation"]["llamacpp"]
        with LlamaServer(lc["server_bin"], ensure_gguf(g["gguf_repo"], g["gguf_pattern"]), lc["port"],
                         lc["parallel"], lc["ctx_per_slot"], lc["n_gpu_layers"], cfg["generation"]["seed"],
                         str(check_dir / "llama_server.log"), lib_path=lc.get("lib_path")):
            ev.start_trial(yaml_path, skip_validation=True)
    else:
        ev.start_trial(yaml_path, skip_validation=True)
    trial = json.loads((check_dir / "trial.json").read_text())[-1]["trial_name"]
    gen = pd.read_parquet(check_dir / trial / "post_retrieve_node_line" / "generator" / "0.parquet")
    autorag_out = pd.DataFrame({"qid": qa["qid"].values, "autorag_text": gen["generated_texts"].values})

    # passages that reached the prompt in AutoRAG's run
    from ragroute.config import RagConfig
    from ragroute.harness.run_stages import NODE_DIR, node_col

    c = RagConfig.from_id(args.config_id)
    if c.rerank:
        ids = pd.read_parquet(check_dir / trial / "post_retrieve_node_line" / "passage_reranker" / "0.parquet")["retrieved_ids"]
    else:
        ids_df = pd.read_parquet(check_dir / trial / "retrieve_node_line" / NODE_DIR[c.retriever] / "0.parquet")
        ids = node_col(ids_df, "retrieved_ids", c.retriever)
    autorag_out["autorag_ids"] = ids.map(list).values

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
