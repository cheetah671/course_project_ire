"""Router B: cheap post-retrieval signals from a probe = BM25 top-pool + dense top-pool.

Probe cost (charged to Router B, see cost.router_overhead): 2 index queries (the dense one includes the
question embedding). No reranker scores and no LLM calls are used here; those would not be "cheap".

Per retriever r in {bm25, dense}:
  r_s1, r_s2, r_s10        raw scores at ranks 1, 2, 10 (BM25 scores are not comparable across queries,
                           so we also give scale-free versions below)
  r_gap12_rel              (s1 - s2) / |s1|
  r_gap1k_rel              (s1 - s10) / |s1|
  r_std10_rel              std(s1..s10) / |s1|
  r_entropy10              entropy of p_i = (s_i - min_20 + eps) / sum over top-10 (scale-invariant)
  r_lex_top1, r_lex_top3   share of query content words found in the top-1 passage / best of top-3
  r_distinct_titles10      distinct article titles in top-10 (multi-hop questions spread over articles)
  r_title_in_query3        a top-3 passage title appears in the question
Cross-retriever:
  x_jaccard3, x_jaccard10  overlap of BM25 and dense id sets
  x_top1_agree             same top-1 passage
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ragroute.eval.metrics import normalize_answer

STOP = set("""a an the of in on at to for from by with and or is are was were be been being which who whom whose
what when where why how did do does that this these those it its as into than then there their his her he she they
""".split())
EPS = 1e-6


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def _scores_feats(prefix: str, s: np.ndarray) -> dict:
    s = np.asarray(s, dtype=float)
    top = s[:10]
    s1 = top[0] if len(top) else 0.0
    denom = abs(s1) + EPS
    shifted = top - s.min() + EPS
    p = shifted / shifted.sum()
    return {
        f"{prefix}_s1": s1,
        f"{prefix}_s2": top[1] if len(top) > 1 else 0.0,
        f"{prefix}_s10": top[-1] if len(top) else 0.0,
        f"{prefix}_gap12_rel": (top[0] - top[1]) / denom if len(top) > 1 else 0.0,
        f"{prefix}_gap1k_rel": (top[0] - top[-1]) / denom if len(top) > 1 else 0.0,
        f"{prefix}_std10_rel": top.std() / denom,
        f"{prefix}_entropy10": float(-(p * np.log(p)).sum()),
    }


def _text_feats(prefix: str, query: str, contents: list[str]) -> dict:
    qw = content_words(query)
    nq = max(len(qw), 1)
    lex = [len(qw & content_words(c)) / nq for c in contents[:3]]
    titles = [c.split("\n", 1)[0] for c in contents[:10]]
    nquery = f" {normalize_answer(query)} "
    return {
        f"{prefix}_lex_top1": lex[0] if lex else 0.0,
        f"{prefix}_lex_top3": max(lex) if lex else 0.0,
        f"{prefix}_distinct_titles10": len(set(titles)),
        f"{prefix}_title_in_query3": int(any(f" {normalize_answer(t)} " in nquery for t in titles[:3] if normalize_answer(t))),
    }


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def build_retrieval_features(queries: pd.DataFrame, bm25: pd.DataFrame, dense: pd.DataFrame) -> pd.DataFrame:
    """queries: qid, query. bm25/dense: qid, ids, scores, contents (the cached top-pool lists)."""
    b = bm25.set_index("qid")
    d = dense.set_index("qid")
    rows = []
    for qid, query in zip(queries["qid"], queries["query"]):
        f = {"qid": qid}
        for prefix, src in (("bm25", b), ("dense", d)):
            row = src.loc[qid]
            f.update(_scores_feats(prefix, row["scores"]))
            f.update(_text_feats(prefix, query, list(row["contents"])))
        bi, di = list(b.loc[qid, "ids"]), list(d.loc[qid, "ids"])
        f["x_jaccard3"] = _jaccard(bi[:3], di[:3])
        f["x_jaccard10"] = _jaccard(bi[:10], di[:10])
        f["x_top1_agree"] = int(bool(bi) and bool(di) and bi[0] == di[0])
        rows.append(f)
    return pd.DataFrame(rows)
