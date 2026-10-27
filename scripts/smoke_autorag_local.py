"""Local CPU smoke test of the AutoRAG harness (no GPU, no vLLM). Run before the Ada pilot.

Builds a tiny DPR-like corpus from HotpotQA's own paragraphs (split into 100-word passages, "title\\ntext"),
then runs the REAL staged pipeline pieces:
  1. AutoRAG Evaluator.start_trial(retrieval_stage.yaml): BM25 + Chroma(bge-base) ingest, bm25/dense/hybrid top-20
  2. AutoRAG SentenceTransformerReranker on each pool
  3. prefix check: AutoRAG's own top-k retrieval (k=3) == first 3 of the cached top-20 pool (bm25, dense, hybrid)
  4. Router-B retrieval features on the cached pools
  5. ChatFstring prompts + Qwen2.5 chat-template token counts (tokenizer only) -> real tokens/passage for budget.py
Generation is skipped (needs vLLM + GPU).

Usage: python scripts/smoke_autorag_local.py --hotpot-parquet <validation parquet> [--n-questions 40 --n-context 400]
Writes everything under data/smoke/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ragroute.config import REPO_ROOT, RagConfig, load_experiment
from ragroute.features.retrieval_features import build_retrieval_features
from ragroute.harness import make_autorag_yaml as may
from ragroute.harness import run_stages as rs


def chunk_words(text: str, n: int = 100):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [""]


def build_tiny(hotpot: pd.DataFrame, n_q: int, n_ctx: int, seed: int, out: Path):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(hotpot), size=n_ctx, replace=False)
    ctx_rows = hotpot.iloc[idx]
    passages, title_to_ids = {}, {}
    for row in ctx_rows.itertuples():
        for title, sents in zip(row.context["title"], row.context["sentences"]):
            if title in title_to_ids:
                continue
            ids = []
            for chunk in chunk_words(" ".join(sents)):
                pid = f"h{len(passages)}"
                passages[pid] = (title, chunk)
                ids.append(pid)
            title_to_ids[title] = ids
    corpus = pd.DataFrame({
        "doc_id": list(passages),
        "contents": [f"{t}\n{c}" for t, c in passages.values()],
        "metadata": [{"title": t, "last_modified_datetime": datetime(2018, 12, 20)} for t, _ in passages.values()],
    })
    qa_rows = ctx_rows.head(n_q)
    qa = pd.DataFrame({
        "qid": [f"hotpotqa:{x}" for x in qa_rows["id"]],
        "query": qa_rows["question"].tolist(),
        "retrieval_gt": [[title_to_ids[t][:2] for t in sorted(set(sf["title"])) if t in title_to_ids] or [[]]
                         for sf in qa_rows["supporting_facts"]],
        "generation_gt": [[a] for a in qa_rows["answer"]],
    })
    meta = pd.DataFrame({"qid": qa["qid"], "dataset": "hotpotqa", "task": "qa", "hops": "multi",
                         "qtype": qa_rows["type"].tolist(), "query": qa["query"]})
    (out / "autorag").mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out / "autorag" / "corpus.parquet", index=False)
    qa.to_parquet(out / "autorag" / "qa.parquet", index=False)
    meta.to_parquet(out / "qa_meta.parquet", index=False)
    return corpus, qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot-parquet", required=True)
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-context", type=int, default=400)
    args = ap.parse_args()

    smoke = REPO_ROOT / "data" / "smoke"
    cfg = load_experiment()
    cfg["paths"].update(data_root=str(smoke), project_dir=str(smoke / "autorag_project"),
                        cache_dir=str(smoke / "cache"), results_dir=str(smoke / "results"),
                        autorag_yaml_dir=str(smoke / "autorag_yaml"))
    report = {}
    hotpot = pd.read_parquet(args.hotpot_parquet)
    corpus, qa = build_tiny(hotpot, args.n_questions, args.n_context, cfg["seed"], smoke)
    report["corpus_passages"] = len(corpus)
    report["queries"] = len(qa)
    report["passage_words_mean"] = float(corpus["contents"].str.split().map(len).mean())
    print(f"tiny corpus: {len(corpus)} passages, {len(qa)} queries")

    # 1. YAMLs + AutoRAG retrieval trial
    ydir = Path(cfg["paths"]["autorag_yaml_dir"])
    (ydir / "full").mkdir(parents=True, exist_ok=True)
    (ydir / "retrieval_stage.yaml").write_text(yaml.safe_dump(may.retrieval_stage_yaml(cfg), sort_keys=False))
    t0 = time.time()
    rs.stage_retrieval(cfg, force=True)
    report["retrieval_stage_seconds_cpu"] = round(time.time() - t0, 1)

    cache = Path(cfg["paths"]["cache_dir"])
    pools = {r: pd.read_parquet(cache / "retrieval" / f"{r}.parquet") for r in rs.RETRIEVERS}
    report["pool_sizes"] = {r: sorted(set(p["ids"].map(len))) for r, p in pools.items()}

    # gold recall@20 per retriever (sanity: retrievers work at all)
    gt = dict(zip(qa["qid"], qa["retrieval_gt"]))
    def recall(p, k):
        hits = []
        for qid, ids in zip(p["qid"], p["ids"]):
            groups = [set(g) for g in gt[qid] if len(g)]
            if groups:
                hits.append(np.mean([bool(g & set(list(ids)[:k])) for g in groups]))
        return round(float(np.mean(hits)), 3)
    report["gold_recall@3"] = {r: recall(p, 3) for r, p in pools.items()}
    report["gold_recall@20"] = {r: recall(p, 20) for r, p in pools.items()}

    # 2. rerank
    t0 = time.time()
    rs.stage_rerank(cfg, force=True)
    report["rerank_seconds_cpu"] = round(time.time() - t0, 1)
    rr = {r: pd.read_parquet(cache / "rerank" / f"{r}.parquet") for r in rs.RETRIEVERS}
    report["rerank_same_id_sets"] = {r: bool(all(set(a) == set(b) for a, b in zip(rr[r]["ids"], pools[r]["ids"])))
                                     for r in rs.RETRIEVERS}
    report["rerank_gold_recall@3"] = {r: recall(p, 3) for r, p in rr.items()}

    # 3. prefix check against AutoRAG's own top-3 retrieval (one trial per retriever, retrieval node line only)
    from autorag.evaluator import Evaluator
    prefix_ok = {}
    for r in rs.RETRIEVERS:
        y = {"vectordb": may.vectordb_block(cfg),
             "node_lines": [{"node_line_name": "retrieve_node_line", "nodes": may.retrieval_nodes(r, 3, cfg)}]}
        yp = ydir / f"prefix_{r}.yaml"
        yp.write_text(yaml.safe_dump(y, sort_keys=False))
        project = Path(cfg["paths"]["project_dir"])
        ev = Evaluator(str(smoke / "autorag" / "qa.parquet"), str(smoke / "autorag" / "corpus.parquet"), str(project))
        ev.start_trial(str(yp))
        trial = json.loads((project / "trial.json").read_text())[-1]["trial_name"]
        df = pd.read_parquet(project / trial / "retrieve_node_line" / rs.NODE_DIR[r] / "0.parquet")
        own = df[f"retrieved_ids{rs.SUFFIX[r]}"].map(list).tolist()
        cached = pools[r]["ids"].map(lambda x: list(x)[:3]).tolist()
        prefix_ok[r] = float(np.mean([a == b for a, b in zip(own, cached)]))
    report["top3_equals_prefix_of_top20"] = prefix_ok

    # 4. Router-B features
    rf = build_retrieval_features(qa[["qid", "query"]], pools["bm25"], pools["dense"])
    report["retrieval_features_shape"] = list(rf.shape)
    report["retrieval_features_nan"] = int(rf.isna().sum().sum())

    # 5. prompts + token counts
    from autorag.nodes.promptmaker import ChatFstring
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["models"]["generators"]["qwen1p5b"]["hf_id"])
    pm = ChatFstring(str(Path(cfg["paths"]["project_dir"])))
    tokens = {}
    for k in (3, 10):
        ctx = rs.contexts_for(cache, "hybrid", k, True)
        df = qa.merge(ctx, on="qid")
        prompts = pm.pure(df, prompt=cfg["prompts"]["qa"])["prompts"]
        n = [len(tok.apply_chat_template(p, add_generation_prompt=True, tokenize=True)) for p in prompts]
        tokens[f"k{k}"] = {"mean": round(float(np.mean(n)), 1), "max": int(np.max(n))}
    passage_tok = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in corpus["contents"].head(500)]
    report["prompt_tokens"] = tokens
    report["tokens_per_passage_mean"] = round(float(np.mean(passage_tok)), 1)
    report["example_prompt"] = prompts.iloc[0]
    out = smoke / "smoke_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "example_prompt"}, indent=2, default=str))


if __name__ == "__main__":
    main()
