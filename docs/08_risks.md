# Deliverable 7: Risk register

L = likelihood, I = impact (H/M/L). "Signal" is how we notice early.

| # | Risk | L | I | Signal | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R1 | **Router-target label imbalance**: one or two configs are oracle-best for most queries; many queries all-fail or all-succeed | H | H | T3 label stats: top-label share, all-fail/all-succeed shares | Regression on all 24 outcomes instead of argmax classification (D20); report label stats; per-dataset static baseline; frame a small gain as a finding | P3 |
| R2 | **Oracle inflated by noise** (max over 24 noisy outcomes) | H | H | Routers recover little of the oracle gap on test (T4) | Treat the oracle as a ceiling only; claim predictability only from held-out routers; report label statistics. (An oracle-level null was tried and dropped: docs/02 D16) | P2 |
| R3 | **EM strictness**: correct verbose answers score 0 | H | M | Human study: EM false-negative rate | F1 as router target; short-answer prompt; answer extraction identical for all configs; "contains" only as a diagnostic | P2 |
| R4 | **Hybrid fusion choices** change hybrid's ranking (RRF k, pool depth, CC vs RRF) | M | M | Hybrid ≈ max(BM25, dense) or worse than both | Fixed RRF k = 60 over a fixed pool of 20 (no tuning on test); optional `hybrid_cc` check on the pilot; stated limitation | P1 |
| R5 | **Train/test leakage** through shared supporting articles (HotpotQA), duplicate questions | M | H | `leakage_report.json`: shared non-hub titles (must be 0), hub share | Grouped split by connected components; script fails on leakage or a giant component; report hub overlap | P2 |
| R6 | **Judge noise** larger than config differences; self-preference | H | M | 150-item κ / Spearman below threshold | Judge never drives routing; pre-registered go/no-go; non-Qwen judge; RAGAS only on a subsample | P2 |
| R7 | **Cost-weight sensitivity**: conclusions flip with weights | M | H | T8 Kendall τ low; frontier changes | Generator-specific measured weights; named alternatives + 500 random draws; report which conclusions are robust | P1 |
| R8 | **Metric-normalisation artifacts** (3 tasks, 3 metrics: QA F1, FEVER accuracy with a 0.5 chance floor, zsRE EM; pooled means mix scales) | H | M | Pooled result driven by one task | Native [0,1] per-query scores, no per-config rescaling; per-task + per-dataset + macro over datasets + macro over tasks; with/without FEVER; chance-corrected FEVER; per-task routers alongside the global one | P2 |
| R9 | **Ada queue and time limits** (2-GPU jobs queue; jobs killed at wall time) | H | M | Pending times > 1 day; killed jobs | Unit-level caching + resume; pilot first; 1.5B before 7B; AWQ fallback for 7B on one GPU (reported) | P1 |
| R10 | AutoRAG staging differs from its end-to-end semantics / version drift | M | H | Equivalence check < threshold | Pin 0.3.27; `check_equivalence.py` on 2 configs; read source on any mismatch | P1 |
| R11 | vLLM / fp16 problems on Turing GPUs (unsupported version, NaN or garbage output) | M | H | Empty or repeated outputs in the pilot | Pin a smoke-tested vLLM in [0.11, 0.31); fall back to an older vLLM or Ampere nodes; log GPU type | P1 |
| R12 | Sub-corpus bias (easier than full Wikipedia; favours dense without dense mining) | H | M | Very high retrieval recall in the pilot | Mined BM25 negatives; optional dense mining; state direction of bias; results framed as relative | P1 |
| R13 | Gold mapping fails for HotpotQA/FEVER titles (2017 vs 2018 snapshots) | M | L | `corpus_stats.json` mapping rate | Keep unmapped queries (no selection bias); gold used for diagnostics only | P2 |
| R14 | Router collapses to config main effects (= best static) | M | M | Router curve overlaps static frontier; `D_only` ≈ full model | Interaction features; GBDT; inspect per-query pick diversity | P3 |
| R15 | Small test set → wide CIs | H | M | Pilot variance → expected CI width | Raise n per dataset (generation is cheap); stratified bootstrap; fewer, pre-declared comparisons | P2 |
| R16 | Dataset identity explains most of routing | M | M | Per-dataset static ≈ router | Report it as a finding; `A_noDS`/`B_noDS` ablations; per-dataset analysis | P3 |
| R17 | HF dataset ids/fields or script-based loaders changed (e.g. KILT) | M | L | prepare_qa errors | `--fever-jsonl` fallback; print examples; zsRE fields checked on the HF validation split 30 Oct | P2 |
| R18 | **Cost weights wrong** (placeholders, or measured on another engine/GPU than the full run) | M | H | `collect` warns "PLACEHOLDER"; `cost_weight_set` column; calibration R² < 0.95 | Job K on the full run's engine and GPU before the full run; weight-sensitivity table T8; components reported separately | P1 |
| R19 | **Per-task test sets small** (~200 queries each for FEVER and zsRE) | H | M | Wide per-task CIs | Report per-task CIs honestly; the pooled ALL block is the headline; raise n for a task if CIs are uninformative | P3 |
| R18 | Latency numbers polluted by co-located jobs | M | L | Large p95/p50 ratios | Exclusive node if allowed; one session; repeats; secondary metric only | P1 |
