# Deliverable 6: Week-by-week plan and 3-person split

Confirmed dates: **mid-submission Fri 2026-10-31**, **final deadline Fri 2026-11-21** (about 3.5 weeks).
Ada limits (docs/04): 1 GPU, 1 job, 6 h per job, so GPU work is a queue of sequential 6-hour jobs.

Scope (revised 30 Oct): 3 tasks over 5 datasets: open-domain QA (NQ, HotpotQA, TriviaQA), fact
verification (FEVER) and slot filling (zsRE); routers trained globally and per task. Cuts forced by the dates:
no dense mining; latency study optional; RAGAS on ~50 test questions per dataset after the 150-item validation.
Cost weights are measured (job K) before the full run.

## Roles
| Person | Owns | Main files |
|---|---|---|
| **P1: Pipeline & compute** | Ada setup, corpus/mining, AutoRAG harness, generation runs, equivalence/repro checks, cost calibration, offline cost, latency study; later the cost-model analysis (T8, T9, T10, A1) | `data/mine_*`, `data/build_corpus.py`, `harness/*`, `cost.py`, `eval/latency.py`, `slurm/*` |
| **P2: Evaluation & statistics** | Dataset loading/sampling, splits + leakage, metrics, bootstrap, judge validation (runs annotation, adjudicates), RAGAS, static/oracle tables (T1–T3, T6, T7, T11, T12) | `data/prepare_qa.py`, `data/splits.py`, `eval/metrics.py`, `eval/judge_validation.py`, `eval/ragas_eval.py`, `stats/*`, `analysis/tables.py` |
| **P3: Router** | Features, oracle analysis, router models, baselines, λ selection, ablations, figures (T4, T5, F2–F6) | `features/*`, `oracle.py`, `router/*`, `analysis/plots.py` |
| **All** | Related work (each reads ~2 core papers and does the reference-list / cited-by check for them), 150-item annotation (P1 and P3 annotate; P2 adjudicates), writing | `docs/01_proposal.md` |

## Mid-submission (Fri 31 Oct)
**Must have (independent of the GPU question)**
1. Proposal + design (docs 01–02); verified citations.
2. Code repo + tests; local AutoRAG validation (top-k = prefix of top-20 exactly; bugs found and fixed).
3. Real data: 1,000 questions per dataset sampled; grouped split + leakage report (0 shared articles train/test).
4. Corpus built on Ada: DPR download, full-corpus BM25 mining, sub-corpus.
5. Budget with measured token counts; Ada constraints and the adjusted plan.

**Should have (needs a vLLM-capable GPU, or the 1080 Ti fallback, by ~29 Oct)**
6. Pilot grid: 100 questions/dataset × 24 configs → first static table, first oracle-vs-best-static headroom, Pareto plot.

**Stretch:** Router A on the pilot. **Fallback if no GPU by 29 Oct:** retrieval-only pilot (BM25/dense/hybrid recall) + items 1–5.

| Day | Ada / pipeline (P1) | Team |
|---|---|---|
| Tue 28 Oct | env install; prepare_qa on Ada; CPU job: download corpus + BM25 index to node scratch, mine candidates; rewrite SLURM scripts for 1 job / 1 GPU / 6 h | Send TA message; read core papers |
| Wed 29 Oct | 100k pilot sub-corpus, splits, AutoRAG YAMLs; GPU job: retrieval + rerank | Decide 2080 Ti vs 1080 Ti fallback from the TA answer |
| Thu 30 Oct | GPU job: pilot generation (both generators, 24 configs); collect; first tables + figures | Draft the mid report |
| Fri 31 Oct | Fill report with pilot numbers | Submit |

## 1–21 Nov (final)
| Dates | P1 (pipeline) | P2 (evaluation) | P3 (router) | Exit criterion |
|---|---|---|---|---|
| 1–7 Nov | 1M sub-corpus; full retrieval/rerank; generation for 3 datasets × 1,000 (≈ 4 sequential 6 h GPU jobs) | collect → oracle table; T2, T3; sample the 150 judge items | Retrieval features; Router A/B on the 1.5B half-grid as soon as it exists | Full grid complete, no gaps |
| 8–14 Nov | Cost calibration; cost-weight reruns (T8, T9) | Annotation + adjudication; judge validation; RAGAS on ~50 q/dataset if it passes | All routers + baselines, 5 seeds, λ selection; ablations T5 | Headline T4 + ablations |
| 15–19 Nov | Appendix (equivalence, environment); optional latency | Breakdowns T6, best-per-type T7 (Holm); check every claim has a CI | Error analysis; final figures F2–F7 | Full draft |
| 20–21 Nov | Buffer, review, submit | | | Submitted |

Cut order if behind: latency study → MLP router → RAGAS at scale (keep the 150-item validation) → TriviaQA (3rd dataset of the QA task) → n per dataset 1,000 → 500. **The three tasks are never cut.**
