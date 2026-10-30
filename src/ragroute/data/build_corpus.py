"""Step 3: build the AutoRAG corpus.parquet (sub-corpus of DPR Wikipedia) and qa.parquet.

Chunking is FIXED: we keep DPR's own 100-word passages (2018-12-20 dump, psgs_w100.tsv, columns
id/text/title) unchanged. contents = "{title}\n{text}" (DPR convention of prepending the title).

Sub-corpus = union of
  (1) full-corpus BM25 top-`bm25_mining_depth` for every sampled query      (mined_bm25.parquet)
  (2) full-corpus dense top-`dense_mining_depth` (only if mined_dense.parquet exists)
  (3) the first `article_neighbours` passages of every gold title (HotpotQA supporting titles, FEVER / zsRE
      KILT provenance titles)
  (4) uniform random passages (seeded) until `subcorpus_size` is reached

Gold passages (AutoRAG retrieval_gt, outer list = AND groups, inner list = OR):
  hotpotqa : one group per supporting title = first 2 passages of that article (intro paragraph)
  fever    : one OR group = first 2 passages of every provenance title
  zsre     : same as fever (KILT provenance = the subject's article)
  nq/trivia: one OR group = mined passages whose normalised text contains a normalised gold answer
             (DPR "has_answer"-style heuristic; title excluded from matching)
  no match : [[]] (kept, flagged has_gold=False; we do NOT drop them, to avoid selection bias)
Gold passages are used only for retrieval diagnostics and leakage grouping, never as router features.

Outputs in {data_root}:
  autorag/corpus.parquet   doc_id, contents, metadata{title, dpr_id, last_modified_datetime}
  autorag/qa.parquet       qid, query, retrieval_gt, generation_gt     (exactly AutoRAG's schema)
  qa_meta.parquet          qid, dataset, task, hops, qtype, answers, gold_titles, has_gold, n_gold
  corpus_stats.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ragroute.config import load_experiment
from ragroute.eval.metrics import normalize_answer

CHUNK_ROWS = 500_000
INTRO_PASSAGES = 2


def norm_title(t: str) -> str:
    """One key for Wikipedia titles from both sources. HotpotQA (2017 dump) and DPR (Dec 2018 dump) differ in
    Unicode form (decomposed vs composed accents, e.g. "Václav Havel") and HotpotQA keeps HTML entities
    ("Army &amp; Navy Stores"). Before this fix 9 of the 200 pilot HotpotQA gold titles were missed (Ada job 44382)."""
    return unicodedata.normalize("NFC", html.unescape(t.replace("_", " "))).strip()


def title_key(s: pd.Series) -> pd.Series:
    return s.map(norm_title)


def _reader(tsv_path, usecols=None):
    return pd.read_csv(tsv_path, sep="\t", chunksize=CHUNK_ROWS, usecols=usecols,
                       dtype={"id": np.int64, "text": str, "title": str}, keep_default_na=False)


def pass1_titles(tsv_path: str, need_titles: set[str], cap: int):
    """Returns all passage ids (int64 array) and, for needed titles, their first `cap` passage ids in order."""
    all_ids, title_to_ids = [], defaultdict(list)
    for chunk in _reader(tsv_path, usecols=["id", "title"]):
        all_ids.append(chunk["id"].to_numpy())
        tk = title_key(chunk["title"])
        mask = tk.isin(need_titles).to_numpy()
        for t, i in zip(tk[mask], chunk["id"].to_numpy()[mask]):
            if len(title_to_ids[t]) < cap:
                title_to_ids[t].append(int(i))
    return np.concatenate(all_ids), title_to_ids


def _answer_patterns(answers):
    pats = []
    for a in answers:
        na = normalize_answer(a)
        if na:
            pats.append(re.compile(r"(?:^| )" + re.escape(na) + r"(?: |$)"))
    return pats


def build(cfg: dict) -> None:
    root = Path(cfg["paths"]["data_root"])
    c = cfg["corpus"]
    rng = np.random.default_rng(cfg["seed"])
    qa = pd.read_parquet(root / "qa_raw.parquet")
    mined = [pd.read_parquet(root / "mined_bm25.parquet")]
    if (root / "mined_dense.parquet").exists():
        mined.append(pd.read_parquet(root / "mined_dense.parquet"))
    mined = pd.concat(mined)
    stats = {"mined_sources": len(mined), "n_queries": len(qa)}

    # ---- pass 1: titles -> passage ids
    need = {norm_title(t) for ts in qa["gold_titles"] for t in ts}
    cap = max(c["article_neighbours"], INTRO_PASSAGES)
    all_ids, title_to_ids = pass1_titles(c["tsv_path"], need, cap)
    stats["full_corpus_passages"] = int(len(all_ids))
    stats["gold_titles_needed"] = len(need)
    stats["gold_titles_found"] = sum(1 for t in need if t in title_to_ids)

    keep = set(mined["dpr_id"].astype(int).tolist())
    n_mined = len(keep)
    for t in need:
        keep.update(title_to_ids.get(t, [])[: c["article_neighbours"]])
        keep.update(title_to_ids.get(t, [])[:INTRO_PASSAGES])
    n_before_fill = len(keep)
    fill = c["subcorpus_size"] - len(keep)
    if fill > 0:
        cand = rng.choice(all_ids, size=min(len(all_ids), fill + fill // 5 + 1000), replace=False)
        cand = cand[~np.isin(cand, np.fromiter(keep, dtype=np.int64))][:fill]
        keep.update(cand.tolist())
    else:
        print(f"WARNING: mined+gold passages ({len(keep)}) exceed subcorpus_size; no random fill")
    stats.update(n_mined=n_mined, n_gold_neighbours=n_before_fill - n_mined, n_random_fill=len(keep) - n_before_fill,
                 subcorpus_passages=len(keep))
    keep_arr = np.sort(np.fromiter(keep, dtype=np.int64))

    # ---- pass 2: write corpus, compute has_answer gold for nq/trivia
    single = qa[qa["dataset"].isin(["nq", "triviaqa"])]
    cand_map = defaultdict(list)
    single_qids = set(single["qid"])
    for qid, dpr_id in zip(mined["qid"], mined["dpr_id"]):
        if qid in single_qids:
            cand_map[int(dpr_id)].append(qid)
    patterns = {q: _answer_patterns(a) for q, a in zip(single["qid"], single["answers"])}
    has_answer = defaultdict(list)
    passage_title = {}

    out_dir = root / "autorag"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_dt = datetime.fromisoformat(c["fixed_datetime"])
    prefix = c["doc_id_prefix"]
    schema = pa.schema([
        ("doc_id", pa.string()), ("contents", pa.string()),
        ("metadata", pa.struct([("title", pa.string()), ("dpr_id", pa.int64()),
                                ("last_modified_datetime", pa.timestamp("us"))])),
    ])
    writer = pq.ParquetWriter(out_dir / "corpus.parquet", schema)
    written = 0
    for chunk in _reader(c["tsv_path"]):
        ids = chunk["id"].to_numpy()
        pos = np.searchsorted(keep_arr, ids)
        pos[pos >= len(keep_arr)] = 0
        sub = chunk[keep_arr[pos] == ids]
        if sub.empty:
            continue
        for dpr_id, text, title in zip(sub["id"], sub["text"], sub["title"]):
            qids = cand_map.get(int(dpr_id))
            if qids:
                nt = normalize_answer(text)
                for q in qids:
                    if any(p.search(nt) for p in patterns[q]):
                        has_answer[q].append(int(dpr_id))
                        passage_title[int(dpr_id)] = title
        table = pa.table({
            "doc_id": [f"{prefix}{i}" for i in sub["id"]],
            "contents": (sub["title"] + "\n" + sub["text"]).tolist(),
            "metadata": [{"title": t, "dpr_id": int(i), "last_modified_datetime": fixed_dt}
                         for i, t in zip(sub["id"], sub["title"])],
        }, schema=schema)
        writer.write_table(table)
        written += len(sub)
    writer.close()
    stats["corpus_rows_written"] = written

    # ---- qa.parquet (AutoRAG schema) + qa_meta.parquet
    ret_gt, gold_titles_out = [], []
    for row in qa.itertuples():
        if row.dataset in ("nq", "triviaqa"):
            ids = has_answer.get(row.qid, [])[:20]
            groups = [[f"{prefix}{i}" for i in ids]] if ids else [[]]
            titles = list(dict.fromkeys(passage_title[i] for i in ids))[:3]
        elif row.dataset == "hotpotqa":
            groups = [[f"{prefix}{i}" for i in title_to_ids.get(norm_title(t), [])[:INTRO_PASSAGES]]
                      for t in row.gold_titles]
            groups = [g for g in groups if g] or [[]]
            titles = list(row.gold_titles)
        else:  # fever, zsre (KILT provenance titles)
            ids = [i for t in row.gold_titles for i in title_to_ids.get(norm_title(t), [])[:INTRO_PASSAGES]]
            groups = [[f"{prefix}{i}" for i in ids]] if ids else [[]]
            titles = list(row.gold_titles)
        ret_gt.append(groups)
        gold_titles_out.append(titles)

    qa_autorag = pd.DataFrame({
        "qid": qa["qid"], "query": qa["query"], "retrieval_gt": ret_gt,
        "generation_gt": qa["answers"].map(list),
    })
    qa_autorag.to_parquet(out_dir / "qa.parquet", index=False)
    meta = qa.drop(columns=["gold_titles"]).assign(
        gold_titles=gold_titles_out,
        n_gold=[sum(len(g) for g in gs) for gs in ret_gt],
    )
    meta["has_gold"] = meta["n_gold"] > 0
    meta.to_parquet(root / "qa_meta.parquet", index=False)
    stats["has_gold_rate"] = meta.groupby("dataset")["has_gold"].mean().round(4).to_dict()
    (root / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build(load_experiment(args.config))


if __name__ == "__main__":
    main()
