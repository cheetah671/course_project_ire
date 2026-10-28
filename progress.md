# Progress report (as of Wed 28 Oct 2026)

**Project:** Which RAG Configuration for Which Query? Cost-Aware Per-Query Configuration Routing (IRE)
**Deadlines:** mid-submission Fri 31 Oct 2026 · final Fri 21 Nov 2026

**In short:** the technical work for the mid-submission is complete. The full pipeline ran end to end on
Ada at pilot scale (300 questions × 24 configurations), with validity checks, results tables, figures and
a written report. What remains for 31 Oct is the team's part: names, submission format, a read-through,
and pushing. After that, the main job is the full run (3,000 questions on a 1M-passage corpus) and the
final analysis.

---

## 1. Done: up to the mid-submission

### 1.1 Design and documentation
| Item | Where |
|---|---|
| Proposal: problem, research questions RQ1–RQ5, hypotheses H1–H4, related work | docs/01_proposal.md |
| Experiment design with decision records D1–D24 | docs/02_experiment_design.md |
| Repo and harness design, Ada/Slurm setup, plan and team split, risk register, requirements checklist | docs/03–09 |
| All 25 references checked against arXiv, ACL Anthology or DOI records (8 core papers, 17 dataset/model/tool references; 2 corrections made); software cited with the exact versions run | docs/01_proposal.md |
| Mid-submission report with pilot results, two Pareto figures and validity checks | docs/10_mid_submission_report.md |

### 1.2 Code (`src/ragroute/`)
- **Data and corpus:** dataset loading and stratified sampling, full-corpus BM25 mining (Pyserini),
  sub-corpus builder, grouped train/dev/test split with a leakage report.
- **Harness:** a staged, cached runner on top of AutoRAG, so every one of the 24 configurations is
  actually run. AutoRAG on its own keeps only the best module after each step.
- **Two generation backends:** llama.cpp (used for the pilot) and vLLM (for the full run), switched by
  one setting.
- **Analysis:** metrics (EM, F1), cost model, features, oracle, routers (logistic regression and GBDT),
  baselines, paired bootstrap, tables and plots.
- **Checks:** a generation comparison tool (backend vs backend, run vs rerun) and an equivalence check
  against AutoRAG's own end-to-end pipeline.
- **Tests:** 26 unit tests, all passing.

### 1.3 Data and corpus
- 1,000 questions sampled per dataset (NQ-open, HotpotQA, TriviaQA); pilot uses 100 per dataset.
- DPR Wikipedia (21,015,324 passages) and its Pyserini BM25 index downloaded to Ada and checked by MD5.
- Pilot sub-corpus: 100,000 passages (each question's full-corpus BM25 top-100, gold-article passages,
  random fill).
- Split 60/20/20 per dataset, grouped by gold article: 0 test questions share an article with training.

### 1.4 Runs on Ada (all at pilot scale, 28 Oct)
| Job | Ada ID | Hardware | What it did |
|---|---|---|---|
| A | 43657 | CPU | Built llama.cpp; downloaded DPR and the BM25 index; mined candidates; built the sub-corpus, split and AutoRAG configs |
| B | 43945 | 1 GPU (GTX 1080 Ti) | All 24 configs: retrieval, reranking, features, generation with both Qwen models, scoring (about 2 h) |
| C | 44053 | CPU | Routers, baselines, oracle, results tables and figures |
| D | 44097 | 1 GPU | Validity checks (below) |
| E | 44382 | CPU | Investigated 32 HotpotQA articles missing from DPR, which led to a title-matching fix |

Two earlier runs failed on bugs, both fixed: 43929 (Chroma database limits) and 44080 (AutoRAG crashing
on llama.cpp's responses).

### 1.5 Pilot results (300 questions; router test split of 60)
| Result | Value |
|---|---|
| Best single configuration (chosen on train): hybrid retrieval, k=10, no rerank, 7B | test F1 0.591 at cost 1,390 |
| Oracle (best configuration per question) | F1 0.787 at cost 163 |
| Headroom (oracle minus best static) | +0.196 F1, 95% CI [0.117, 0.280] |
| Headroom by dataset | HotpotQA +0.31, NQ +0.26, TriviaQA +0.01 |
| Routers | none beats the best static config on quality (all CIs include 0); several are within the CI at 25–66% lower cost |
| Main effects | the 7B model adds ~0.09 F1 for ~6× the cost; dense retrieval and reranking matter more than k |

At pilot size these results show that the pipeline works. They are not findings.

### 1.6 Validity checks (job D)
| Check | Result |
|---|---|
| Retrieved passages, our staged runner vs AutoRAG's own pipeline | 100% identical (2 configs) |
| Generated answers, same comparison | 100% and 94% identical |
| Our input-token counts vs the serving engine's | 100% identical |
| Same configuration rerun | 97.3% identical answers (parallel batching on llama.cpp) |

### 1.7 Bugs found and fixed (highlights)
- AutoRAG node files use unsuffixed column names; AutoRAG breaks on pandas 3; its validator crashes on
  rerun.
- The transformers-5 chat template returns a dict; Chroma's SQL-variable and batch-size limits; AutoRAG
  crashing on llama.cpp's responses.
- HotpotQA titles not matching DPR (Unicode accent forms, HTML entities): 9 of the 32 misses were ours,
  fixed with a test.
- Dropped an invalid significance test for the oracle (its null scored 0.9999, above the real 0.787).
  Predictability is now judged only by held-out routers (decision D16).

---

## 2. Started early for the full run: vLLM on RTX 2080 Ti (28 Oct)
The TA confirmed RTX 2080 Ti nodes (gnode042 onwards). This work is **not needed for the mid-submission**.

- **Set up:** a separate Ada environment (`ragroute-vllm`) with vLLM 0.11.2, chosen because the nodes'
  NVIDIA driver supports up to CUDA 12.8. Install script `slurm/ada/setup_vllm_env.sh`, test job
  `slurm/ada/jobV_vllm_smoke_gpu.sbatch`.
- **Memory settings:** tuned for 11 GB after two out-of-memory failures.
- **Smoke test results** (jobs 45057 and 45063; 2 configs × 300 pilot questions per model):

| | 1.5B model | 7B model (4-bit AWQ) |
|---|---|---|
| Speed vs pilot (llama.cpp on the 1080 Ti) | 3.3–4.7× faster | 3.5–3.8× faster |
| Rerun gives identical answers | 100% | 100% |
| Token counts match ours | 100% | 100% |
| Same answer text as the pilot | 39–40% | 70–77% |
| F1, vLLM vs pilot | 0.369 vs 0.393; 0.427 vs 0.453 | 0.485 vs 0.486; 0.522 vs 0.536 |

- **Open issue:** under vLLM the 1.5B model often answers in full sentences, and at least one 7B answer
  came out garbled. The likely cause is half precision (fp16) on this GPU, which has no bf16. This has to
  be resolved before choosing the engine for the full run (see §4).

---

## 3. Left for the mid-submission (by Fri 31 Oct)
| Task | Who |
|---|---|
| Add team names and roll numbers (report line 4) | Team |
| Check the required submission format (PDF, repo link, other); convert the report if needed | Team (Claude can convert) |
| Read through the report | Team |
| Commit and push the remaining changes | Team |

---

## 4. Left after the mid-submission (1–21 Nov)

### 4.1 First: choose the generation engine (about 1 day)
- Rerun the 1.5B model in vLLM at full 32-bit precision to test whether fp16 causes the differences.
- Count garbled 7B answers across all smoke-test outputs.
- Decide: vLLM (about 3.5× faster, no rerun noise) or llama.cpp on the 2080 Ti (matches the pilot,
  slower, ~3% rerun noise).
- Pin the chosen vLLM version in `requirements-cluster.txt` and record the decision in docs/02.

### 4.2 Full run and analysis
| Dates | Work |
|---|---|
| 1–7 Nov | Build the ~1M-passage sub-corpus; retrieval and reranking for 3,000 questions; generation for all 24 configs (about 4 sequential 6 h GPU jobs); oracle table; retrieval features |
| 8–14 Nov | Cost calibration and cost-weight sensitivity; 150-item human annotation to validate the LLM judge; Routers A and B with 5 seeds; choosing the cost-quality trade-off on dev; ablations |
| 15–19 Nov | Breakdowns by dataset and question type (Holm-corrected); error analysis; where routing does not help; final figures |
| 20–21 Nov | Review and submit the final report |

**Also planned:** re-measure rerun noise for the 7B model on the chosen engine.

**Cut order if behind:** latency study → MLP router → RAGAS at scale (keep the 150-item validation) →
TriviaQA → 1,000 to 500 questions per dataset.

### 4.3 Risks to watch
- **Headroom may not be predictable.** No pilot router beat the best static config on quality. The full
  run has about 5× more training data, and the proposal's fallback framing covers a negative result.
- **Ada limits:** one 6 h job at a time. Every stage caches its work and resumes after a timeout.
- **Ada home disk:** 3.9 GB free after the vLLM install. Large downloads already go to each node's local
  disk.
