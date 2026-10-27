"""Optional step 2b: mine dense candidates over the FULL corpus with the same embedding model.

Only run when corpus.dense_mining = true. Without it, the sub-corpus contains BM25's full-corpus
hard negatives but not dense's, which makes dense retrieval look easier than on full Wikipedia
(a bias in favour of dense / hybrid). Cost: one embedding pass over ~21M passages (see docs/budget).

  embed : python -m ragroute.data.mine_dense embed --shard I --num-shards S   (SLURM array over I)
  search: python -m ragroute.data.mine_dense search --num-shards S
Outputs: {cache}/dense_full/shard_I.{ids.npy,emb.npy}  and  {data_root}/mined_dense.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ragroute.config import load_experiment

CHUNK_ROWS = 200_000


def embed_shard(cfg, shard: int, num_shards: int, batch: int):
    from sentence_transformers import SentenceTransformer

    out_dir = Path(cfg["paths"]["cache_dir"]) / "dense_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(cfg["models"]["embedding"], device="cuda")
    model.half()
    ids, embs = [], []
    reader = pd.read_csv(cfg["corpus"]["tsv_path"], sep="\t", chunksize=CHUNK_ROWS,
                         dtype={"id": np.int64, "text": str, "title": str}, keep_default_na=False)
    for ci, chunk in enumerate(reader):
        if ci % num_shards != shard:
            continue
        texts = (chunk["title"] + "\n" + chunk["text"]).tolist()
        e = model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        ids.append(chunk["id"].to_numpy())
        embs.append(e.astype(np.float16))
        print(f"shard {shard}: chunk {ci} done")
    np.save(out_dir / f"shard_{shard}.ids.npy", np.concatenate(ids))
    np.save(out_dir / f"shard_{shard}.emb.npy", np.concatenate(embs))


def search(cfg, num_shards: int, block: int = 500_000):
    import torch
    from sentence_transformers import SentenceTransformer

    root = Path(cfg["paths"]["data_root"])
    depth = cfg["corpus"]["dense_mining_depth"]
    qa = pd.read_parquet(root / "qa_raw.parquet")
    model = SentenceTransformer(cfg["models"]["embedding"], device="cuda")
    q = model.encode([cfg["models"]["embedding_query_instruction"] + x for x in qa["query"]],
                     normalize_embeddings=True, convert_to_tensor=True).half()
    best_s = torch.full((len(qa), depth), -1e4, device="cuda", dtype=torch.float32)
    best_i = torch.zeros((len(qa), depth), device="cuda", dtype=torch.int64)
    dense_dir = Path(cfg["paths"]["cache_dir"]) / "dense_full"
    for s in range(num_shards):
        ids = torch.from_numpy(np.load(dense_dir / f"shard_{s}.ids.npy")).cuda()
        emb = np.load(dense_dir / f"shard_{s}.emb.npy", mmap_mode="r")
        for start in range(0, len(emb), block):
            e = torch.from_numpy(np.ascontiguousarray(emb[start:start + block])).cuda()
            scores = (q @ e.T).float()
            top_s, top_j = scores.topk(min(depth, scores.shape[1]), dim=1)
            cand_s = torch.cat([best_s, top_s], dim=1)
            cand_i = torch.cat([best_i, ids[start:start + block][top_j]], dim=1)
            keep_s, keep_j = cand_s.topk(depth, dim=1)
            best_s, best_i = keep_s, torch.gather(cand_i, 1, keep_j)
        print(f"searched shard {s}")
    rows = [
        {"qid": qid, "rank": r, "dpr_id": int(i), "score": float(sc)}
        for qid, ii, ss in zip(qa["qid"], best_i.cpu().numpy(), best_s.cpu().numpy())
        for r, (i, sc) in enumerate(zip(ii, ss))
    ]
    pd.DataFrame(rows).to_parquet(root / "mined_dense.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "search"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    if args.cmd == "embed":
        embed_shard(cfg, args.shard, args.num_shards, args.batch)
    else:
        search(cfg, args.num_shards)


if __name__ == "__main__":
    main()
