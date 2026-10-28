import json

import pandas as pd

from ragroute.harness.compare_generations import compare


def _gen(texts, in_tok):
    return pd.DataFrame({"qid": ["q1", "q2"], "generated_text": texts, "input_tokens": in_tok, "output_tokens": [2, 3]})


def test_compare_two_caches(tmp_path):
    cache = tmp_path / "cache"
    for sub in ("generation", "generation_vllm"):
        (cache / sub).mkdir(parents=True)
    pd.DataFrame({"qid": ["q1", "q2"], "task": ["qa", "qa"], "answers": [["Paris"], ["1990"]]}).to_parquet(
        tmp_path / "qa_meta.parquet")
    _gen(["Paris", "1990"], [10, 20]).to_parquet(cache / "generation" / "c1.parquet")
    _gen(["Paris", "1991"], [10, 20]).to_parquet(cache / "generation_vllm" / "c1.parquet")
    _gen(["x", "y"], [1, 1]).to_parquet(cache / "generation_vllm" / "only_in_b.parquet")  # no reference: skipped
    timings = [{"stage": "generate_c1", "seconds": 9.0},                                  # pilot (untagged)
               {"stage": "generate_c1", "seconds": 7.0},                                  # later untagged rerun
               {"stage": "generate_c1", "subdir": "generation_vllm", "seconds": 3.0}]
    (cache / "stage_timings.jsonl").write_text("\n".join(json.dumps(t) for t in timings))
    cfg = {"paths": {"data_root": str(tmp_path), "cache_dir": str(cache)}}

    (row,) = compare(cfg, "generation", "generation_vllm")
    assert row["config_id"] == "c1" and row["n"] == 2
    assert row["same_text_rate"] == 0.5
    assert row["quality_generation"] == 1.0 and row["quality_generation_vllm"] == 0.5
    assert row["input_tokens_equal_rate"] == 1.0
    assert row["seconds_generation"] == 9.0 and row["seconds_generation_vllm"] == 3.0
