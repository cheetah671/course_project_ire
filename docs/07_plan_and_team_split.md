# Deliverable 6: Week-by-week plan and 3-person split

Assumes 8 weeks from Mon 2026-10-26 (VERIFY deadline, open item 3). AutoRAG plumbing is budgeted at
**one week**; weeks 2–7 go to oracle labelling, the router and evaluation.

## Roles
| Person | Owns | Main files |
|---|---|---|
| **P1: Pipeline & compute** | Ada setup, corpus/mining, AutoRAG harness, generation runs, equivalence/repro checks, cost calibration, offline cost, latency study; later the cost-model analysis (T8, T9, T10, A1) | `data/mine_*`, `data/build_corpus.py`, `harness/*`, `cost.py`, `eval/latency.py`, `slurm/*` |
| **P2: Evaluation & statistics** | Dataset loading/sampling, splits + leakage, metrics, bootstrap, judge validation (runs annotation, adjudicates), RAGAS, static/oracle tables (T1–T3b, T6, T7, T11, T12) | `data/prepare_qa.py`, `data/splits.py`, `eval/metrics.py`, `eval/judge_validation.py`, `eval/ragas_eval.py`, `stats/*`, `analysis/tables.py` |
| **P3: Router** | Features, oracle analysis, router models, baselines, λ selection, ablations, figures (T4, T5, F2–F6) | `features/*`, `oracle.py`, `router/*`, `analysis/plots.py` |
| **All** | Related work (each reads ~2 core papers and does the reference-list / cited-by check for them), 150-item annotation (P1 and P3 annotate; P2 adjudicates), writing | `docs/01_proposal.md` |

## Weeks
| Week | Dates | P1 (pipeline) | P2 (evaluation) | P3 (router) | Exit criterion |
|---|---|---|---|---|---|
| 1 | Oct 26 – Nov 1 | Envs on Ada, prefetch, **pilot** (100k sub-corpus, 100 q/dataset): retrieval, rerank, both generators; `04_checks` | prepare_qa for all datasets (print examples, VERIFY fields/sizes); metrics + tests; splits on pilot | Features + router code on synthetic data (`python -m ragroute.synthetic`); feature unit checks | Equivalence ≥ [[0.95]] agreement; measured throughput → `budget.py` and `cost_weights.yaml` updated |
| 2 | Nov 2 – 8 | Full mining + 1M sub-corpus; full retrieval/rerank; start generation (1.5B first) | Leakage report on full sample; decide FEVER from the budget; expected CI widths from pilot variance | Retrieval features on full pools; router dry-run on the 1.5B half-grid | Retrieval caches complete; FEVER in/out decided |
| 3 | Nov 9 – 15 | 7B generation; rerun failed configs; offline cost record | `collect` → oracle table; T2, T3, T3b; sample 150 items; annotation starts | Oracle analysis: label stats, F4; first Router A/B | Full grid complete (24 × N cells, no gaps) |
| 4 | Nov 16 – 22 | Latency study (one session); cost calibration final | Annotation + adjudication; judge validation run (`07_judge MODE=validate`); decision on RAGAS | All models × feature sets × 5 seeds; baselines incl. Adaptive-RAG-style; λ selection | Headline T4 draft; judge go/no-go |
| 5 | Nov 23 – 29 | Cost-weight reruns (named sets), amortisation (T8, T9) | RAGAS on subsample (if go); breakdowns T6, best-per-type T7 with Holm | Ablations T5, importance F6, no-reuse accounting; F2/F3 final | All tables generated from code |
| 6 | Nov 30 – Dec 6 | Appendix A1, environment table, limitations | Statistics review of every claim (CI attached?) | Error analysis: queries where routers fail vs oracle | Draft sections 3–5 |
| 7 | Dec 7 – 13 | Write §3, §8 limitations | Write §4, §7 | Write §5–6 | Full draft; resolve every VERIFY |
| 8 | Dec 14 – 20 | Buffer: reruns, polish, final review, submission | | | Submitted |

Cut order if behind: latency study → FEVER → MLP router → dense mining → RAGAS at scale (keep the 150-item validation).
