"""Step 2: mine hard-negative / candidate passages from the FULL Wikipedia corpus.

Why: AutoRAG's BM25 is rank_bm25.BM25Okapi over an in-memory pickled token list, and the AutoRAG corpus
is loaded fully into pandas by every retrieval module, so the full 21M-passage DPR corpus is not
practical inside AutoRAG. We therefore build a ~1M-passage sub-corpus (build_corpus.py) that contains
each query's full-corpus BM25 top-100 (and optionally dense top-100) plus random fill. This keeps the
hard negatives each retriever would have faced, while staying within AutoRAG's limits.

BM25 mining uses a Pyserini prebuilt Lucene index of the same DPR passages (no indexing needed).
VERIFY: index name (configs: corpus.pyserini_index) and that hit.docid equals the DPR passage id.
Pyserini needs a JDK (21 for recent versions; VERIFY on Ada: `module avail java`).

Output: {data_root}/mined_bm25.parquet with columns qid, rank, dpr_id, score
Usage:  python -m ragroute.data.mine_candidates [--threads 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ragroute.config import load_experiment


def mine_bm25(qa: pd.DataFrame, index_name: str, depth: int, threads: int, index_dir: str | None = None) -> pd.DataFrame:
    """index_dir: an already-unpacked Lucene index (preferred on Ada: Pyserini's own downloader cannot resume an
    interrupted 9 GB download, so job A fetches it with `wget -c`, checks the MD5, and unpacks it)."""
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_dir) if index_dir else LuceneSearcher.from_prebuilt_index(index_name)
    print(f"index: {index_dir or index_name}, documents: {searcher.num_docs}")
    rows = []
    qids, queries = qa["qid"].tolist(), qa["query"].tolist()
    step = 1000
    for start in range(0, len(qids), step):
        hits = searcher.batch_search(queries[start:start + step], qids[start:start + step], k=depth, threads=threads)
        for qid, hs in hits.items():
            rows.extend({"qid": qid, "rank": r, "dpr_id": int(h.docid), "score": float(h.score)} for r, h in enumerate(hs))
        print(f"mined {min(start + step, len(qids))}/{len(qids)}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--index-dir", default=None, help="unpacked Lucene index directory (skips Pyserini's downloader)")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    root = Path(cfg["paths"]["data_root"])
    qa = pd.read_parquet(root / "qa_raw.parquet")
    out = mine_bm25(qa, cfg["corpus"]["pyserini_index"], cfg["corpus"]["bm25_mining_depth"], args.threads,
                    args.index_dir)
    out.to_parquet(root / "mined_bm25.parquet", index=False)
    print(f"wrote {len(out)} rows; queries with 0 hits: {qa['qid'].nunique() - out['qid'].nunique()}")


if __name__ == "__main__":
    main()
