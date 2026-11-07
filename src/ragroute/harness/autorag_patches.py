"""Small runtime patches for AutoRAG 0.3.27's Chroma wrapper (autorag/vectordb/chroma.py).

Found on the 100k pilot corpus (Ada job 43929):
1. Chroma.is_exist() asks Chroma for all corpus ids in one query -> SQLite "too many SQL variables".
   It also tests membership against a list (quadratic in corpus size).
2. Chroma.add_embedding() writes all embeddings in one collection.add(), which exceeds Chroma's
   maximum batch size on large corpora.
Both are replaced with batched versions. Results are identical (same ids, same embeddings), only chunked.
Found on the 500k full-run corpus (Ada job 63979, 7 Nov 2026):
3. utils.preprocess.validate_qa_from_corpus_dataset() builds a boolean mask over the whole corpus for EVERY gold id
   (tens of thousands of full scans at 5,000 queries x 500k passages: >10 min of one CPU core in Evaluator.__init__).
   Replaced with a set lookup that raises the same AssertionError on the same inputs.
Call apply() before constructing an Evaluator or starting any AutoRAG trial.
"""
from __future__ import annotations

from typing import List

ID_BATCH = 5000  # well under SQLite's variable limit (32,766 on SQLite >= 3.32)


def _max_batch(store) -> int:
    try:
        return int(store.client.get_max_batch_size())
    except Exception:  # older/other clients: a conservative size that works everywhere
        return 5000


async def _is_exist(self, ids: List[str]) -> List[bool]:
    from chromadb.api.models.AsyncCollection import AsyncCollection

    existed = set()
    for i in range(0, len(ids), ID_BATCH):
        chunk = ids[i:i + ID_BATCH]
        if isinstance(self.collection, AsyncCollection):
            res = await self.collection.get(chunk, include=[])
        else:
            res = self.collection.get(chunk, include=[])
        existed.update(res["ids"])
    return [x in existed for x in ids]


def _add_embedding(self, ids: List[str], embeddings) -> None:
    step = _max_batch(self)
    for i in range(0, len(ids), step):
        self.collection.add(ids=ids[i:i + step], embeddings=embeddings[i:i + step])


def _validate_qa_from_corpus_dataset(qa_df, corpus_df) -> None:
    """Same checks as AutoRAG 0.3.27's version (which ids are collected, same assertion), O(total ids)."""
    import numpy as np

    qa_ids = []
    for retrieval_gt in qa_df["retrieval_gt"].tolist():
        if isinstance(retrieval_gt, list) and (retrieval_gt[0] != [] or any(bool(g) is True for g in retrieval_gt)):
            for gt in retrieval_gt:
                qa_ids.extend(gt)
        elif isinstance(retrieval_gt, np.ndarray) and retrieval_gt[0].size > 0:
            for gt in retrieval_gt:
                qa_ids.extend(gt)
    corpus_ids = set(corpus_df["doc_id"].tolist())
    no_exist_ids = [i for i in qa_ids if i not in corpus_ids]
    assert len(no_exist_ids) == 0, f"{len(no_exist_ids)} doc_ids in retrieval_gt do not exist in corpus_df."


def apply() -> None:
    import autorag.evaluator
    import autorag.utils.preprocess
    from autorag.vectordb.chroma import Chroma

    # module globals are looked up at call time, so patching the evaluator's imported name is enough
    autorag.utils.preprocess.validate_qa_from_corpus_dataset = _validate_qa_from_corpus_dataset
    if hasattr(autorag.evaluator, "validate_qa_from_corpus_dataset"):
        autorag.evaluator.validate_qa_from_corpus_dataset = _validate_qa_from_corpus_dataset
    if getattr(Chroma, "_ragroute_patched", False):
        return
    Chroma.is_exist = _is_exist
    Chroma.add_embedding = _add_embedding
    Chroma._ragroute_patched = True
