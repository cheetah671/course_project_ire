"""Secondary metrics with RAGAS (faithfulness, answer relevancy) on a SUBSAMPLE.

Run in a separate environment (env_ragas; see requirements-ragas.txt): RAGAS pulls LangChain versions
that conflict with AutoRAG's. The judge LLM is served by vLLM's OpenAI-compatible server
(`vllm serve <judge> --port 8001`), and must not be a Qwen model (the generators are Qwen; a judge from the
same family risks self-preference bias).

API used (legacy, present in ragas 0.2.x; still importable in 0.4.x with deprecation warnings; VERIFY on
the pinned version): ragas.evaluate, EvaluationDataset, metrics Faithfulness and ResponseRelevancy,
LangchainLLMWrapper, LangchainEmbeddingsWrapper.

Short-answer caveat (write-up): with one-phrase answers, faithfulness reduces to "is this one claim
entailed by the passages" and answer relevancy is close to saturated. We still report them, but only
after the 150-item human validation shows they track human judgements.

Usage: python -m ragroute.eval.ragas_eval [--n-per-dataset 100] [--items items.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment


# RAGAS judges a free-text answer against passages. FEVER answers are a bare SUPPORTS/REFUTES label, for which
# faithfulness and answer relevancy are not meaningful, so verification is scored by accuracy only.
JUDGE_TASKS = ("qa", "slot_filling")


def build_items(cfg: dict, n_per_dataset: int) -> pd.DataFrame:
    """Test-split subsample: n queries per dataset x ALL 24 configs, so any router's choice is covered."""
    res = Path(cfg["paths"]["results_dir"])
    runs = pd.read_parquet(res / "runs_scored.parquet")
    qa = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "qa.parquet")[["qid", "query", "generation_gt"]]
    test = runs[(runs["split"] == "test") & runs["task"].isin(JUDGE_TASKS)]
    rng = np.random.default_rng(cfg["seed"])
    qids = []
    for ds, g in test.drop_duplicates("qid").groupby("dataset"):
        u = g["qid"].to_numpy()
        qids += list(rng.choice(u, size=min(n_per_dataset, len(u)), replace=False))
    items = test[test["qid"].isin(qids)].merge(qa, on="qid")
    if "retrieved_ids" not in items.columns:  # runs_scored.parquet already carries them (merging again gave _x/_y)
        cache = Path(cfg["paths"]["cache_dir"]) / "generation"
        ctx = pd.concat([pd.read_parquet(cache / f"{c}.parquet", columns=["qid", "config_id", "retrieved_ids"])
                         for c in items["config_id"].unique()])
        items = items.merge(ctx, on=["qid", "config_id"])
    corpus = pd.read_parquet(Path(cfg["paths"]["project_dir"]) / "data" / "corpus.parquet", columns=["doc_id", "contents"])
    lookup = dict(zip(corpus["doc_id"], corpus["contents"]))
    items["contexts"] = items["retrieved_ids"].map(lambda ids: [lookup[i] for i in ids])
    return items


def run_ragas(items: pd.DataFrame, judge_model: str, base_url: str, emb_model: str, workers: int = 16) -> pd.DataFrame:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, base_url=base_url, api_key="EMPTY", temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=emb_model))
    ds = EvaluationDataset.from_list([
        {"user_input": q, "retrieved_contexts": list(c), "response": p, "reference": list(g)[0]}
        for q, c, p, g in zip(items["query"], items["contexts"], items["pred"], items["generation_gt"])
    ])
    kwargs = {}
    try:  # token accounting for the judge budget (VERIFY name in pinned version)
        from ragas.cost import get_token_usage_for_openai
        kwargs["token_usage_parser"] = get_token_usage_for_openai
    except ImportError:
        pass
    result = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy()], llm=llm, embeddings=emb,
                      run_config=RunConfig(max_workers=workers, timeout=300), **kwargs)
    scores = result.to_pandas()
    try:
        usage = result.total_tokens()
        print("judge token usage:", usage)
    except Exception:  # noqa: BLE001
        pass
    keep = [c for c in scores.columns if c in ("faithfulness", "answer_relevancy")]
    return pd.concat([items[["qid", "config_id", "dataset"]].reset_index(drop=True), scores[keep]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-per-dataset", type=int, default=None)
    ap.add_argument("--items", default=None, help="score a prepared item file instead (e.g. the 150 validation items)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=400, help="items per saved chunk (resume unit)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    j = cfg["judge"]
    items = pd.read_parquet(args.items) if args.items else build_items(cfg, args.n_per_dataset or j["ragas_queries_per_dataset"])
    print(f"{len(items)} items to judge")
    out = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "ragas_scores.parquet"
    # Resumable: items are judged in chunks and each finished chunk is saved, so a job stopped by the time limit
    # continues with the next chunk instead of starting over (the test subsample is ~14,000 judge calls).
    chunk_dir = out.parent / f"{out.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    items = items.sort_values(["config_id", "qid"], kind="mergesort").reset_index(drop=True)
    parts = []
    for start in range(0, len(items), args.chunk):
        f = chunk_dir / f"chunk_{start:06d}_{len(items)}.parquet"
        if not f.exists():
            part = run_ragas(items.iloc[start:start + args.chunk].reset_index(drop=True), j["model"], j["base_url"],
                             cfg["models"]["embedding"])
            part.to_parquet(f.with_suffix(".tmp"), index=False)
            f.with_suffix(".tmp").replace(f)
            print(f"judged items {start}-{min(start + args.chunk, len(items))} of {len(items)}", flush=True)
        parts.append(pd.read_parquet(f))
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out, index=False)
    print(scores.groupby("config_id")[["faithfulness", "answer_relevancy"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
