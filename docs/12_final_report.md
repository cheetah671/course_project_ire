# Which RAG Configuration for Which Query? An End-to-End, Cost-Counted Test of Per-Query Configuration Routing

Final report, Information Retrieval and Extraction, November 2026

**Team Agents:** Samyak Jain (2024101062), Arnav Agnihotri (2024101103), Sherley Sonali Peethala (2023111022)

> Status of this draft: method sections are final; every number marked `[[...]]` is filled from the result files
> in `results/` (tables T1–T12, figures F2–F9) once the full run on Ada completes. Never replace one with a guess.

## Abstract
[[Written last, from the headline table T4. Use version W (routers beat the best static configuration) or
version S (they do not) from docs/01_proposal.md, whichever the confidence intervals support.]]

## 1. Introduction
A retrieval-augmented generation (RAG) system makes several choices for every question: which retriever to use,
how many passages to read, whether to rerank them, and which language model writes the answer. Benchmark studies
(Wang et al., 2024) and tuning tools (AutoRAG; AutoRAG-HP; Orbach et al., 2025; HAMMER, 2026) search for one good
configuration per dataset. But a configuration that is best on average may be wasteful for easy questions and
too weak for hard ones.

We run every one of 24 configurations on every question, which tells us, per question, which configurations
answer correctly and what each one costs. This turns a configuration sweep into labels for a prediction problem:
can the right configuration be predicted for a new question, before running all of them, and how much quality or
cost does that save over always using the best fixed configuration?

Two recent papers frame the question. BRANE (Pan et al., 2026) routes each query to a pipeline configuration by
predicting correctness per configuration and penalising cost, and reports large savings. RCT-MARS (Srivastava,
2026) describes a *routing paradox*: the per-query oracle beats every fixed retrieval method by a wide margin,
yet learned routers do not beat the best fixed one. Our study tests which of these two pictures holds for a full
RAG pipeline (retrieval, reranking and generation), with the router's own cost counted, across three tasks.

**Contributions.**
1. A fully crossed, cost-instrumented grid: 24 configurations × 5,000 questions from five datasets and three
   tasks, with per-question quality and countable cost (tokens, LLM calls, reranker pairs, index queries).
2. The oracle ceiling against the best static configuration, with label statistics that show where routing has
   room at all.
3. Routers that see only the question (Router A) or also cheap post-retrieval signals (Router B, which are
   query-performance predictors in the IR sense), trained globally and per task, with the cost of their signals
   charged, against static, per-dataset, per-task and Adaptive-RAG-style baselines.
4. A cost model that separates once-per-corpus (offline) cost from per-query (marginal) cost, with measured
   weights and a sensitivity analysis.

## 2. Related work
| Line of work | Representative papers | Relation to this project |
|---|---|---|
| One configuration per dataset | Wang et al. 2024 (EMNLP); AutoRAG (Kim et al. 2024); AutoRAG-HP (Fu et al. 2024); Orbach et al. 2025; Barker et al. 2025; HAMMER (Zhou et al. 2026) | Static search; we ask whether the choice should vary per question. AutoRAG is our harness. |
| Per-query adaptation | Adaptive-RAG (Jeong et al. 2024); METIS (Ray et al. 2025); Su et al. 2025; RAGRouter (Zhang et al. 2025); RAGRouter-Bench (Wang et al. 2026) | Adaptive-RAG is re-implemented as a baseline. METIS adapts under serving load; we study offline, countable cost. |
| Per-query configuration routing | **BRANE** (Pan et al. 2026); **RCT-MARS** (Srivastava 2026) | BRANE has the same cost-penalised formulation (we do not claim it). RCT-MARS's routing paradox is the hypothesis we test end to end. |
| Query performance prediction (QPP) | Clarity (Cronen-Townsend et al. 2002); WIG (Zhou & Croft 2007); NQC (Shtok et al. 2012); QPP for RAG query variants (Arabzadeh et al. 2026) | Router B's retrieval-score signals are post-retrieval QPP predictors; Arabzadeh et al. find cheap pre-retrieval predictors often suffice. |
| Evaluation of adaptive retrieval and of RAG | Qian et al. 2026 (budget-aware evaluation); RAGAS (Es et al. 2024) | We report frontiers and realised cost; RAGAS is used only after human validation. |

How the literature was found: we pulled all 909 papers citing the six core papers from OpenAlex and screened them
by hand (docs/11_literature_update.md), as the TA asked after our proposal missed the 2024 papers.

## 3. Setup

### 3.1 Tasks, datasets and splits
Three tasks over one Wikipedia corpus, 1,000 sampled questions per dataset (5,000 in all):

| Task | Datasets | Primary metric (router target) | Also reported |
|---|---|---|---|
| Open-domain QA | NQ-open (single-hop), HotpotQA (multi-hop: bridge 799, comparison 201), TriviaQA | token F1 | EM |
| Fact verification | FEVER (KILT; 497 SUPPORTS, 503 REFUTES) | accuracy | chance-corrected 2·acc − 1 |
| Slot filling | zsRE (KILT; "subject \| relation" → value) | exact match | token F1 |

Questions are split 60/20/20 into train/dev/test per dataset, **grouped**: questions that share a gold Wikipedia
article are always on the same side (union-find over shared gold titles and near-duplicate questions), so a
router cannot memorise "questions about article X". [[Leakage report: residual test–train title overlap.]]

### 3.2 Corpus
DPR's 100-word passages of the December 2018 English Wikipedia (21,015,324 passages; fixed chunking). AutoRAG's
BM25 scores every passage in Python, so we search a 500,000-passage sub-corpus built from: each question's top-50
passages from BM25 over all 21M passages (our own Lucene index), the first 10 passages of every gold article, and
seeded random passages. [[T1: composition counts.]]

### 3.3 The configuration grid
| Choice | Options |
|---|---|
| Retriever | BM25 (Porter stemmer); dense (`bge-base-en-v1.5`, Chroma, cosine); hybrid (reciprocal rank fusion of BM25 top-20 and dense top-20, k = 60) |
| Passages given to the generator | top-3 or top-10 |
| Reranker | off; or `ms-marco-MiniLM-L-6-v2` cross-encoder re-scoring a pool of 20, keep top-k |
| Generator | Qwen2.5-1.5B-Instruct [[dtype from job B1]]; Qwen2.5-7B-Instruct-AWQ (4-bit) |

Greedy decoding, at most 32 output tokens, one prompt template per task, identical answer post-processing for all
configurations. Generation runs on vLLM 0.11.2 on one RTX 2080 Ti.

**Harness.** AutoRAG keeps only the best module after each pipeline step, which would leave most of the grid
empty. We call its modules stage by stage and cache each stage (retrieve once, rerank once, generate per
generator), so all 24 configurations run on all questions. An equivalence check runs two configurations through
AutoRAG's own end-to-end pipeline and compares retrieved passages and answers [[A1 numbers]].

### 3.4 Metrics, normalisation and statistics
Every per-question score stays on its own [0, 1] scale. Pooled numbers are reported three ways: micro (per
question), macro over datasets, and macro over tasks (each task weighs the same). We did not z-score or min-max
the scores: both change whenever a configuration is added or removed, and per-question min-max is undefined when
all configurations tie, which is common. Because FEVER accuracy starts at 0.5 by chance, we repeat the headline
with FEVER scored as 2·acc − 1 (T3c, T4c).

All comparisons are paired over the same test questions: 10,000 bootstrap resamples, stratified by dataset (or
task), with Holm correction within each family of comparisons. The primary metric was fixed before the run.

### 3.5 Cost model
Per-question (marginal) cost is a weighted sum of exact counts:

cost = a_in[g]·input tokens + a_out[g]·output tokens + b·LLM calls + c·reranker pairs + d·index queries

The counts come from the run logs (our token counts were checked against vLLM's [[A1]]). The weights are GPU
milliseconds measured on the run's own GPU and engine (calibration job K: forced output lengths and prompt sizes,
three repeats, prefix caching off) [[fit R², weights]]. Building the indexes (embedding 500k passages, the BM25
index) is a once-per-corpus cost, reported separately and spread over an explicit number of queries per index
build (10³ to 10⁶; F9). Results are re-checked under three other weight sets and 500 random perturbations
(T8, T8b, F8). Wall-clock latency is not a primary metric.

## 4. The static landscape and the oracle ceiling
[[T2 excerpt: the best configurations per dataset and their cost.]]
[[T3: oracle vs best static per dataset and overall, headroom with CI; label statistics: share of questions where
all configurations fail, all succeed, or they disagree.]]
[[F4: which configuration is oracle-best, by question type.]]

## 5. Routing method
**Formulation.** One model predicts the quality q̂(question, configuration) for all 24 configurations; the
predicted cost ĉ is the configuration's mean training cost within the question's dataset (cost is nearly fixed
given configuration and dataset; its coefficient of variation is in T2). The router picks argmax q̂ − λ·ĉ/s,
with s one fixed cost scale. Sweeping λ traces a cost–quality curve without retraining. (This is the same
formulation as BRANE.) Regression on all 24 outcomes, rather than classifying "the best configuration", avoids
ill-defined labels when many configurations tie.

**Features** (the professor's four groups): (1) query: length, question words, entity and number cues,
comparison and bridge cues, a 32-dimensional PCA of the question embedding, a learned multi-hop score; (2) data:
dataset and task identity; (3) configuration encoding; (4) the observed quality and cost, which are the training
target and never an input. Router B adds a probe: BM25 top-20 and dense top-20 for the question, giving score
levels and gaps, spread, entropy, query–passage word overlap, and BM25–dense agreement.

**Cost of the router.** Router A pays one question-embedding pass unless it picks a dense or hybrid configuration
(which needs that embedding anyway). Router B pays two index queries for the probe; by default the chosen
configuration's own retrieval is served from the probe, and a conservative accounting without this reuse is
also reported.

**Models and training.** Fractional logistic regression with question × configuration interactions, LightGBM, and
a small MLP; five seeds each; trained globally (all tasks) and per task. λ is chosen on dev at (a) iso-cost (best
quality at no more than the best static configuration's cost) and (b) iso-quality (lowest cost at no less than
its quality), then evaluated once on test.

**Baselines.** Best static configuration (chosen on train); the static cost–quality frontier; best static per
dataset; best static per task; Adaptive-RAG-style (a question classifier predicts single-hop / multi-hop /
verification / slot filling, then the best static configuration for that class); cheapest; random; oracle.

## 6. Results
### 6.1 Headline
[[T4 ALL block, iso-cost and iso-quality, for Router A and B (best model), global and per task; F2 Pareto plot.]]

### 6.2 Router A vs Router B, and what the signals are worth
[[T5 ablations: A_noDS, A, A_noEmb, B, B_noDS, D_only, R_only × LR/GBDT/MLP; F6, F7 importance; probe reuse vs
no reuse.]]

### 6.3 Where routing helps and where it does not
[[T6 by dataset, task, hops and HotpotQA type (Holm-adjusted); T7 best configuration per question type.]]

### 6.4 Robustness
[[T8/T8b/F8 cost weights; T9/F9 amortisation; T3c/T4c chance-corrected FEVER; micro vs macro.]]

## 7. LLM-judge metrics
[[T11: human–human and judge–human agreement on 150 hand-labelled dev items; EM false-negative rate. T12 RAGAS
on the test subsample only if the pre-registered rule (κ ≥ 0.6, Spearman CI low > 0.4) passes.]]

## 8. Discussion and limitations
[[What the result means for the routing paradox and for static recipes.]]
Limitations: fixed chunking; a 500k sub-corpus instead of all of Wikipedia (absolute retrieval scores are not
comparable to published numbers, and dense retrieval did not mine its own hard negatives); two generators from
one family, the larger one 4-bit quantised; English Wikipedia only; EM strictness for zsRE; cost weights belong to
one GPU type and engine.

## 9. Conclusion
[[Two or three sentences, from the headline.]]

## Appendix
A1 equivalence and reproducibility · A2 leakage report · A3 environment (GPU, library versions from the job logs) ·
A4 hyper-parameters · A5 annotation guide (docs/05) · A6 design decisions D1–D24 (docs/02).

## References
See docs/01_proposal.md (references 1–41).
