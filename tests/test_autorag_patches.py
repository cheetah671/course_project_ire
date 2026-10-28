"""The batched Chroma id check / insert must work beyond SQLite's variable limit (job 43929 failure)."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("autorag")

from ragroute.harness import autorag_patches  # noqa: E402

N = 40_000  # > 32,766 (SQLite variable limit)


def test_batched_add_and_is_exist(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    store = SimpleNamespace(client=client, collection=client.get_or_create_collection("patchtest"))
    ids = [f"d{i}" for i in range(N)]
    emb = np.random.default_rng(0).random((N, 4), dtype=np.float32)

    autorag_patches._add_embedding(store, ids[: N // 2], emb[: N // 2])
    assert store.collection.count() == N // 2

    flags = asyncio.run(autorag_patches._is_exist(store, ids))
    assert flags == [i < N // 2 for i in range(N)]


def test_apply_patches_autorag_chroma():
    from autorag.vectordb.chroma import Chroma

    autorag_patches.apply()
    assert Chroma.is_exist is autorag_patches._is_exist
    assert Chroma.add_embedding is autorag_patches._add_embedding
