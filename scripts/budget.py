"""Budget arithmetic: configs x queries x (generation tokens + judge tokens) -> GPU-hours (+ API-price reference).

All throughput numbers are PLANNING ASSUMPTIONS, not measurements. Replace them with the week-1 pilot
measurements (stage_timings.jsonl) and re-run:  python scripts/budget.py --prefill-7b 3100 ...
API prices are illustrative placeholders; VERIFY current price sheets before quoting any dollar figure.
"""
from __future__ import annotations

import argparse

DATASETS = {"nq": 1000, "hotpotqa": 1000, "triviaqa": 1000, "fever": 1000, "zsre": 1000}
# Measured on the pilot (Ada job 43945, GTX 1080 Ti, llama.cpp, 300 questions x 12 configs per generator)
PILOT_MIN = {"qwen7b": 73.0, "qwen1p5b": 19.0}
PILOT_QUERIES = 300


def tokens_per_call(k: int, passage_tokens: int, overhead: int) -> int:
    return overhead + k * passage_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage-tokens", type=int, default=147,
                    help="title + 100 words, Qwen tokenizer. Extrapolated from the local AutoRAG smoke test "
                         "(104 tokens per ~73-word passage incl. title, 2026-10-27); re-measure on real DPR passages")
    ap.add_argument("--prompt-overhead", type=int, default=80, help="system+instruction+chat template+question")
    ap.add_argument("--out-tokens", type=float, default=8.0)
    ap.add_argument("--prefill-7b", type=float, default=2500.0, help="tok/s, 7B fp16, TP=2 on 2x RTX 2080 Ti (ASSUMED)")
    ap.add_argument("--prefill-1p5b", type=float, default=10000.0, help="tok/s, 1.5B fp16 on 1 GPU (ASSUMED)")
    ap.add_argument("--gpus-7b", type=int, default=2)
    ap.add_argument("--overhead-factor", type=float, default=1.3, help="model load, decode, stragglers")
    ap.add_argument("--judge-in-tokens", type=int, default=3500, help="RAGAS faithfulness+relevancy per item (ASSUMED)")
    ap.add_argument("--judge-out-tokens", type=int, default=300)
    ap.add_argument("--judge-tps", type=float, default=2500.0, help="7-8B judge on 2 GPUs (ASSUMED)")
    ap.add_argument("--api-in", type=float, default=0.15, help="$ per 1M input tokens (ILLUSTRATIVE, VERIFY)")
    ap.add_argument("--api-out", type=float, default=0.60, help="$ per 1M output tokens (ILLUSTRATIVE, VERIFY)")
    ap.add_argument("--vllm-speedup", type=float, default=3.5, help="vLLM/2080 Ti vs llama.cpp/1080 Ti (smoke test)")
    a = ap.parse_args()

    k_values, retr, rerank = [3, 10], 3, 2
    per_gen_cfgs_per_k = retr * rerank          # 6 configs per (generator, k)
    t3 = tokens_per_call(3, a.passage_tokens, a.prompt_overhead)
    t10 = tokens_per_call(10, a.passage_tokens, a.prompt_overhead)
    print(f"input tokens per call: k=3 -> {t3}, k=10 -> {t10}")
    per_q_per_gen = per_gen_cfgs_per_k * (t3 + t10)  # input tokens per query per generator (12 configs)
    print(f"input tokens per query per generator (12 configs): {per_q_per_gen:,}")
    print()
    print(f"{'datasets':<34}{'queries':>8}{'LLM calls':>11}{'in tok (M)':>11}{'gen GPU-h':>10}{'judge-all GPU-h':>16}{'judge-all $':>12}")
    scopes = (["nq", "hotpotqa"], ["nq", "hotpotqa", "triviaqa"], ["nq", "hotpotqa", "triviaqa", "fever"],
              ["nq", "hotpotqa", "triviaqa", "fever", "zsre"])
    for names in scopes:
        n = sum(DATASETS[d] for d in names)
        calls = n * 24
        tin_gen = n * per_q_per_gen  # per generator
        gpu_h_7b = tin_gen / a.prefill_7b / 3600 * a.gpus_7b
        gpu_h_1p5 = tin_gen / a.prefill_1p5b / 3600
        gen_gpu_h = (gpu_h_7b + gpu_h_1p5) * a.overhead_factor
        judge_in = calls * a.judge_in_tokens
        judge_gpu_h = judge_in / a.judge_tps / 3600 * 2 * a.overhead_factor
        judge_usd = judge_in / 1e6 * a.api_in + calls * a.judge_out_tokens / 1e6 * a.api_out
        print(f"{'+'.join(names):<34}{n:>8,}{calls:>11,}{2 * tin_gen / 1e6:>11.1f}{gen_gpu_h:>10.1f}{judge_gpu_h:>16.1f}{judge_usd:>12.0f}")
    print()
    print("Per extra 1,000-query dataset: +24,000 LLM calls, "
          f"+{2 * 1000 * per_q_per_gen / 1e6:.1f}M input tokens, "
          f"+{(1000 * per_q_per_gen / a.prefill_7b / 3600 * a.gpus_7b + 1000 * per_q_per_gen / a.prefill_1p5b / 3600) * a.overhead_factor:.1f} generation GPU-h")
    for n_sub in (100, 200):
        items = n_sub * 5 * 24
        h = items * a.judge_in_tokens / a.judge_tps / 3600 * 2 * a.overhead_factor
        print(f"RAGAS subsample {n_sub} q/dataset x 24 configs x 5 datasets = {items:,} items -> {h:.1f} judge GPU-h")
    print(f"Judge validation: 150 items -> {150 * a.judge_in_tokens / a.judge_tps / 3600 * 2:.2f} GPU-h")
    print()
    print("MEASURED extrapolation from the pilot generation times (more reliable than the assumptions above):")
    for names in scopes:
        n = sum(DATASETS[d] for d in names)
        h_llama = sum(PILOT_MIN.values()) / 60 * n / PILOT_QUERIES
        print(f"  {'+'.join(names):<34}{n:>6,} q: llama.cpp/1080 Ti {h_llama:5.1f} GPU-h;"
              f" vLLM/2080 Ti (x{a.vllm_speedup} measured in the smoke test) {h_llama / a.vllm_speedup:5.1f} GPU-h")


if __name__ == "__main__":
    main()
