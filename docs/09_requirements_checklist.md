# Deliverable 8: Requirements checklist and open items

Status: ✅ satisfied in the plan/code · 🟡 satisfied with a stated deviation or pending a VERIFY · ⏳ needs cluster results.

## Ground rules
| Requirement | Where | Status |
|---|---|---|
| No fabricated results/sizes/APIs/citations; placeholders; VERIFY marks | All docs use `[[...]]`; AutoRAG/RAGAS APIs checked against package source; sizes and citations marked VERIFY | ✅ |
| Decision → Alternatives → Why → Risk for major decisions | docs/02 D1–D24 | ✅ |
| Explicit assumptions; ≤3 clarifying questions only if blocking | docs/00 §A; none were blocking, so none asked; uncertain items are open items | ✅ |
| No project history | — | ✅ |

## 1. Idea and motivation
| Requirement | Where | Status |
|---|---|---|
| Every config on every query → oracle labelling | docs/02 D6, D19; `harness/run_stages.py`, `collect.py`, `oracle.py` | ✅ |
| RQ; (i) oracle ceiling, (ii) Router A, (iii) Router B, on the Pareto frontier vs best static | docs/01 §3; T3/T4; F2 | ✅ ⏳ |
| Generation-side cost and post-retrieval signals as angles | docs/02 D17, D21–D22 | ✅ |
| Write-up planned for both outcomes | docs/01 abstract W/S; docs/06 §C.9; T3b null test | ✅ |

## 2. Related work
| Requirement | Where | Status |
|---|---|---|
| Position against Wang et al., AutoRAG, AutoRAG-HP, arXiv:2505.03452, Adaptive-RAG, METIS, RAGAS, LegalBench-RAG, chunking-vs-cost | docs/01 §2 | ✅ bibliographic details of all 8 checked 2026-10-27; 🟡 chunking-vs-cost citations TODO |
| Justify running the full grid despite AutoRAG-HP | docs/01 §2 (AutoRAG-HP row); docs/02 D12 | ✅ |
| Reminder: pull reference lists of the first four papers and check who cites them | docs/00 §C.6; docs/01 §2; docs/07 "All" | ✅ |
| Do not invent citations | Only listed works plus original dataset/tool papers, all VERIFY | ✅ |

## 3. Data
| Requirement | Where | Status |
|---|---|---|
| NQ-open + HotpotQA over a stated Wikipedia dump and passage format | docs/02 D1; `configs/experiment.yaml` | ✅ URL and passage count verified |
| 1–2 extra datasets, 2-sentence justification each, same index, compute impact | docs/02 D4 (TriviaQA recommended, FEVER optional, MuSiQue/2Wiki considered) | ✅ |
| Fixed chunking, why, limitation | docs/02 D3 | ✅ |
| ~1,000 queries/dataset; sampling procedure, seed, split by query | docs/02 D5; `data/prepare_qa.py`, `data/splits.py` | ✅ run on the real datasets: 4 × 1,000 sampled, deterministic, 600/200/200 per dataset |
| Leakage check through shared supporting docs | `data/splits.py` (grouped split, fail-fast guards), `leakage_report.json`; tests | ✅ on real data: 293 shared articles found, 0 shared between train and test after grouping |

## 4. Harness, grid, compute
| Requirement | Where | Status |
|---|---|---|
| AutoRAG directly as harness; YAML; parquet prep; commands | docs/03; `make_autorag_yaml.py`, `build_corpus.py`, `run_stages.py` | ✅ 🟡 equivalence check ⏳ |
| What AutoRAG does not do natively + workarounds (VERIFY) | docs/02 D6 table (checked against 0.3.27 source) | ✅ |
| 24-config fully crossed grid with named embedding model, fusion, cross-encoder, pool policy, 2 generators, greedy | docs/02 D7–D10; `configs/experiment.yaml` | ✅ |
| vLLM on Ada; SLURM templates; caching; reproducibility (seeds, versions, GPU type) | `slurm/*`, docs/04 | 🟡 Ada specifics VERIFY |
| Budget arithmetic → GPU-hours + API reference; effect on judge use and dataset count | docs/02 D12; `scripts/budget.py` | ✅ (assumed throughput; re-run after pilot) |

## 5. Metrics and statistics
| Requirement | Where | Status |
|---|---|---|
| Primary EM + token-F1, pre-committed | docs/02 D13; `eval/metrics.py` | ✅ |
| Metric per dataset; normalisation across metrics with justification and pitfalls | docs/02 D14 | ✅ |
| RAGAS secondary; judge validation on ~150 items (correlation, κ); subsample; protocol + spreadsheet | docs/02 D15; docs/05; `annotation/judge_validation_template.csv`; `eval/judge_validation.py`, `eval/ragas_eval.py` | ✅ ⏳ |
| Seeds, variance, 95% CIs, paired bootstrap (10k, code) for every claim | docs/02 D16; `stats/bootstrap.py`; `router/train_eval.py::headline_table` | ✅ |
| Per-question-type best-config claims backed by bootstrap CIs | `best_config_stability`, T7 | ✅ |

## 6. Cost model
| Requirement | Where | Status |
|---|---|---|
| Offline (embedding, index build time/size) vs marginal (index queries, rerank pairs, tokens, calls), reported separately | docs/02 D17; `cost.py`; `scripts/record_offline_cost.py`; T2, T9 | ✅ |
| Amortisation factor + sensitivity | `cost_weights.yaml` (1e3–1e6); T9, F8 | ✅ |
| Scalar cost a·tokens + b·calls + c·rerank + d·index, documented weights, weight sensitivity | `cost.py`, `cost_weights.yaml`, T8 | 🟡 deviation: `a` is generator-specific and split into input/output (reason in D17); the spec's form is one sensitivity set |
| Latency secondary: p50/p95, repeats, randomised order, same node/session only | docs/02 D18; `eval/latency.py`; `slurm/06_latency.sbatch`; `latency_summary` refuses mixed sessions | ✅ |

## 7. Router
| Requirement | Where | Status |
|---|---|---|
| Oracle table (query, config) with normalised quality and cost | `harness/collect.py` → `runs_scored.parquet` | ✅ |
| Feature groups 1–4 | docs/02 D21 (group 4 = target, not input) | 🟡 group 4 reinterpreted as training target (A10) |
| Predict quality and cost, argmax q̂ − λĉ; simpler alternative described; recommendation | docs/02 D20 | ✅ |
| Router A (question only) | feature sets `A`, `A_noDS`, `A_noEmb` | ✅ |
| Router B (+ score distribution, gaps, entropy, BM25-vs-dense overlap, lexical overlap), with its cost charged | `features/retrieval_features.py`; `cost.router_overhead` (reuse and no-reuse) | ✅ |
| Baselines: best static, best static under budget, cheapest, random, oracle, Adaptive-RAG-style; METIS conceptual | docs/02 D23; `router/train_eval.py`; docs/01 §5.4 | ✅ (+ per-dataset static) |
| Held-out test by query; Pareto frontier with all 24 statics + routers; breakdown by dataset and type; paired bootstrap vs best static; ablations (feature groups, model, λ); honest reporting | docs/02 D24; T4–T7; F2, F3, F5 | ✅ ⏳ |

## 8. Deliverables
| # | Deliverable | Where |
|---|---|---|
| 1 | Proposal/report skeleton | docs/01 |
| 2 | Repo + runnable code | docs/03; `src/ragroute/`; `tests/` (19 passing locally) |
| 3 | SLURM scripts | `slurm/`; docs/04 |
| 4 | 150-item annotation guide + template | docs/05; `annotation/` |
| 5 | Tables, figures, report template | docs/06 |
| 6 | Week plan + 3-person split | docs/07 |
| 7 | Risk list | docs/08 |
| 8 | This checklist + open items | docs/09 |
| — | "Bllm" interpretations | docs/00 §B |

## Open items
1. Meaning of "Bllm" (docs/00 §B); base design proceeds without it.
2. Ada: GPU type, account/partition, wall-time limit, scratch paths, `--exclusive`, internet on compute nodes.
3. Deadline and report format.
4. Judge model choice (non-Qwen) and the κ threshold sign-off.
5. FEVER in or out (week-2 budget check).
6. Reference lists and citing papers of Wang et al., AutoRAG, AutoRAG-HP, arXiv:2505.03452; chunking-vs-cost papers. (Bibliographic details of the 8 core papers are now verified.)
7. vLLM version that runs fp16 on Ada GPUs; Qwen2.5 fp16 sanity check. (Floor verified: compute capability ≥ 7.5, so no 1080 Ti.)
8. Pyserini docid format (index name verified).
9. Measured throughput and token counts to replace the planning assumptions in `scripts/budget.py` and `configs/cost_weights.yaml`.
10. ✔ Dataset ids, field names and split sizes verified; loaders run on the real data.
