import numpy as np
import pandas as pd
import pytest

from ragroute.config import RagConfig
from ragroute.eval.latency import retrieval_depth, rrf, sample_questions, summarise_rows


def test_sample_questions_balanced_test_only_and_deterministic():
    meta = pd.DataFrame({"qid": [f"q{i}" for i in range(60)], "dataset": ["a"] * 30 + ["b"] * 30, "task": "qa"})
    splits = pd.DataFrame({"qid": meta["qid"], "split": (["train"] * 20 + ["test"] * 10) * 2})
    s1 = sample_questions(meta, splits, 4, seed=1)
    s2 = sample_questions(meta, splits, 4, seed=1)
    assert s1["dataset"].value_counts().to_dict() == {"a": 4, "b": 4}
    assert set(s1["qid"]) <= set(splits.loc[splits["split"] == "test", "qid"])
    assert s1["qid"].tolist() == s2["qid"].tolist()


def test_rrf_matches_definition():
    # d2 is rank 2 in a and rank 1 in b: 1/62 + 1/61 beats d1 (1/61 only)
    assert rrf(["d1", "d2"], ["d2", "d3"], k=60, depth=3) == ["d2", "d1", "d3"]
    assert rrf(["d1", "d2"], ["d2", "d3"], k=60, depth=1) == ["d2"]


def test_retrieval_depth():
    assert retrieval_depth(RagConfig("bm25", 3, True, "g"), 20) == 20
    assert retrieval_depth(RagConfig("bm25", 3, False, "g"), 20) == 3


def _rows(session="s1"):
    rng = np.random.default_rng(0)
    n = 100
    t = rng.uniform(0.1, 1.0, n)
    return pd.DataFrame({"session_id": session, "host": "h", "gpu": "g", "generator": "qwen1p5b",
                         "config_id": ["bm25-k3-rr0-qwen1p5b"] * 50 + ["dense-k3-rr0-qwen1p5b"] * 50,
                         "t_retrieval": t / 10, "t_rerank": 0.0, "t_generate": t, "t_total": t * 1.1,
                         "output_tokens": 5})


def test_summarise_rows_quantiles_and_session_guard():
    lat = _rows()
    out = summarise_rows(lat).set_index("config_id")
    sub = lat[lat["config_id"] == "bm25-k3-rr0-qwen1p5b"]["t_total"]
    assert out.loc["bm25-k3-rr0-qwen1p5b", "p50_s"] == pytest.approx(sub.quantile(0.5))
    assert out.loc["bm25-k3-rr0-qwen1p5b", "p95_s"] == pytest.approx(sub.quantile(0.95))
    assert (out["n"] == 50).all()
    with pytest.raises(ValueError):
        summarise_rows(pd.concat([_rows("s1"), _rows("s2")]))
