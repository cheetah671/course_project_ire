"""Convert the DPR Wikipedia TSV (id, text, title) into Pyserini JsonCollection shards for our own BM25 index.

Why: the prebuilt Pyserini index (9.2 GB) downloaded at ~0.2 MB/s from Ada on 7 Nov 2026 (~12 h), while the DPR TSV
itself came in 2.5 min. Building the index from the TSV takes well under an hour with 10 threads.
Each document: {"id": "<dpr_id>", "contents": "<title>\\n<text>"}, the same string AutoRAG's corpus uses
(build_corpus.py), so mining and in-corpus BM25 see identical text. hit.docid == DPR id by construction.

Usage: python -m ragroute.data.dpr_to_jsonl --tsv psgs_w100.tsv.gz --out DIR [--shards 10]
Then:  python -m pyserini.index.lucene --collection JsonCollection --input DIR --index IDX \
         --generator DefaultLuceneDocumentGenerator --threads 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_ROWS = 500_000


def convert(tsv: str, out: Path, shards: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    files = [open(out / f"dpr_{i:02d}.jsonl", "w", encoding="utf-8") for i in range(shards)]
    n = 0
    try:
        reader = pd.read_csv(tsv, sep="\t", chunksize=CHUNK_ROWS, dtype={"id": np.int64, "text": str, "title": str},
                             keep_default_na=False)
        for c, chunk in enumerate(reader):
            f = files[c % shards]
            for i, text, title in zip(chunk["id"], chunk["text"], chunk["title"]):
                f.write(json.dumps({"id": str(i), "contents": f"{title}\n{text}"}, ensure_ascii=False) + "\n")
            n += len(chunk)
            print(f"converted {n:,} passages", flush=True)
    finally:
        for f in files:
            f.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=10)
    args = ap.parse_args()
    n = convert(args.tsv, Path(args.out), args.shards)
    (Path(args.out) / "_DONE").write_text(str(n))
    print(f"wrote {n:,} passages to {args.shards} shards")


if __name__ == "__main__":
    main()
