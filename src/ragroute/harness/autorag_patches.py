"""Small runtime patches for AutoRAG 0.3.27's Chroma wrapper (autorag/vectordb/chroma.py).

Found on the 100k pilot corpus (Ada job 43929):
1. Chroma.is_exist() asks Chroma for all corpus ids in one query -> SQLite "too many SQL variables".
   It also tests membership against a list (quadratic in corpus size).
2. Chroma.add_embedding() writes all embeddings in one collection.add(), which exceeds Chroma's
   maximum batch size on large corpora.
Both are replaced with batched versions. Results are identical (same ids, same embeddings), only chunked.
Call apply() before starting any AutoRAG trial.
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


def apply() -> None:
    from autorag.vectordb.chroma import Chroma

    if getattr(Chroma, "_ragroute_patched", False):
        return
    Chroma.is_exist = _is_exist
    Chroma.add_embedding = _add_embedding
    Chroma._ragroute_patched = True
