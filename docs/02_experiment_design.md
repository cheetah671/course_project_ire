# Experiment design (decision records, budget, metrics, statistics, cost model, router)

Format of each record: **Decision → Alternatives considered → Why this one → Risk / limitation.**
Values live in `configs/experiment.yaml`; this document explains them.

---

## 1. Data

### D1. Corpus: DPR Wikipedia passages (2018-12-20 dump, 100-word passages)
- **Decision.** One shared index for all datasets: the DPR Wikipedia split `psgs_w100.tsv`
  (TSV, header `id, text, title`; 21,015,324 passages ✔ (matches the Pyserini `wikipedia-dpr-100w` index
  metadata); URL live ✔, `psgs_w100.tsv.gz` = 4,694,541,059 bytes; record a checksum after download).
  AutoRAG `contents = "{title}\n{text}"`, `doc_id = "w{id}"`.
- **Alternatives.** (a) KILT knowledge source (2019-08-01 snapshot), which ships provenance for NQ, HotpotQA,
  TriviaQA and FEVER on a single snapshot. (b) A separate corpus per dataset (HotpotQA's own 2017
  introductory paragraphs). (c) The datasets' own distractor paragraphs (HotpotQA distractor setting).
- **Why.** DPR passages are the standard open-domain setting for NQ-open and TriviaQA, are already
  segmented (so chunking is genuinely fixed), and one index for all datasets means retriever and cost
  differences are not confounded with corpus differences. (c) makes retrieval trivial; (b) breaks the
  shared index; (a) is the strongest alternative and the fallback if FEVER becomes central.
- **Risk.** HotpotQA (2017 abstracts) and FEVER (2017 claims) were built on other snapshots. Some
  supporting titles are missing or renamed in the 2018 dump, so gold passages are mapped by title
  (`corpus_stats.json` reports the mapping rate) and some answers may have changed. Affects all configs
  equally; reported as a limitation.

### D2. A ~1M-passage sub-corpus instead of all 21M passages
- **Decision.** Sub-corpus = full-corpus BM25 top-100 per query (Pyserini prebuilt index) ∪ the first
  10 passages of every gold title ∪ seeded random passages up to 1,000,000 (pilot: 100,000).
  Optional: full-corpus dense top-100 mined with the same bge model (`dense_mining: true`).
- **Alternatives.** (a) Full 21M inside AutoRAG. (b) Full 21M outside AutoRAG (Pyserini + FAISS).
  (c) Gold + random only.
- **Why.** Checked in AutoRAG 0.3.27's source: its BM25 is `rank_bm25.BM25Okapi` over a pickled,
  fully tokenised corpus held in memory, and every retrieval module loads `corpus.parquet` into pandas.
  At 21M passages that is not workable on an Ada node (memory, and seconds-to-minutes per query). (b)
  would mean rebuilding retrieval outside the harness, which the spec rules out. (c) removes hard
  negatives and makes retrieval unrealistically easy. Mined BM25 negatives keep each query's real
  lexical confusers.
- **Risk.** Absolute retrieval scores are higher than on full Wikipedia and not comparable to published
  numbers. Without dense mining, dense retrieval misses its own hard negatives and is advantaged
  relative to BM25. Mitigation: enable dense mining if budget allows (~+10 GPU-h, one pass); otherwise
  state the direction of the bias. Relative comparisons and routing (the object of study) are less affected.

### D3. Fixed chunking
- **Decision.** No chunking arm. Use DPR's 100-word passages as-is (no further overlap or splitting).
- **Alternatives.** Chunk size {100, 200, 400} × overlap arm on raw article text.
- **Why.** QA benchmarks ship (or are evaluated on) pre-segmented corpora; a chunking arm would mean
  re-segmenting Wikipedia and re-mapping gold evidence, and the answer-bearing unit is already a passage.
  The arm would multiply the grid (×3) while being close to vacuous for these benchmarks.
- **Risk / limitation (state in the report).** Results say nothing about chunking choices, which matter
  for long-document domains (legal, manuals; cf. LegalBench-RAG). Listed as future work.

### D4. Datasets and extra tasks
| Dataset | Role | Justification (two sentences) | Shares index? | Compute impact |
|---|---|---|---|---|
| NQ-open (HF `google-research-datasets/nq_open`, validation, 3,610 ✔) | single-hop | Real user search questions with short answers, the standard open-domain QA benchmark on DPR Wikipedia. Lexically varied, so dense vs BM25 differences are expected. | yes | base |
| HotpotQA (HF `hotpotqa/hotpot_qa`, distractor config, validation 7,405 ✔; 80% bridge / 20% comparison; questions/answers/supporting titles only) | multi-hop | Bridge and comparison questions needing two articles, where k and generator size should matter most. Provides question-type labels for the per-type analysis. | yes (title mapping) | base |
| **TriviaQA** (unfiltered.nocontext, validation 11,313 ✔; ~14 answer aliases per question on average, so EM is more lenient than on NQ) — **recommended extra** | single-hop, different style | Trivia questions written independently of evidence: longer, entity-rich, and known to be friendly to lexical retrieval, so the best retriever plausibly differs from NQ. Uses the same DPR index as in the DPR paper, so it adds query diversity at zero corpus cost. | yes | +1,000 queries: +24,000 LLM calls, +24.9M input tokens, **+4.0 generation GPU-h** |
| **FEVER** (KILT version, 2-way SUPPORTS/REFUTES; validation 10,444 ✔; 205 duplicate claims removed by our filter; sample is ~50/50 SUPPORTS/REFUTES) — optional | non-QA verification | Tests whether one router transfers across task types and exercises the metric normalisation (accuracy vs F1). Claims are short and evidence is single-article, so a cheap config may often suffice. | yes (title mapping; claims from a 2017 snapshot) | +1,000 queries: +24,000 calls, ~+3.5 GPU-h (shorter outputs) |
| MuSiQue / 2WikiMultiHopQA — not chosen | multi-hop | Harder compositional multi-hop questions. Their evidence comes from their own paragraph pools, so sharing the DPR index needs another mapping step. | partly | same per-dataset cost; extra mapping effort |

Decision: base = NQ + HotpotQA; add TriviaQA by default (cheap, same index). FEVER is added after the
week-2 budget check; it is the first item dropped if time is short.

### D5. Sampling and split
- **Sampling** (`data/prepare_qa.py`): drop items without a gold answer and exact duplicate questions;
  `rng = default_rng(20261026 + crc32(dataset))`; HotpotQA stratified by `type`, FEVER by label
  (largest-remainder allocation); n = 1,000 per dataset; output sorted by qid.
- **Decision (split).** Train/dev/test = 60/20/20 **by query**, grouped by shared gold documents
  (`data/splits.py`). Queries are linked if they share a gold title (except "hub" titles shared by more
  than 20 queries) or are duplicate questions (optionally embedding cosine ≥ 0.95); connected components
  are assigned whole, balanced per dataset. The script fails if any non-hub title crosses train/test,
  and fails if one component exceeds 5% of queries.
- **Alternatives.** Random split; grouped 5-fold cross-fitting (every query gets an out-of-fold prediction).
- **Why.** Random splits leak through HotpotQA's shared supporting articles: the router could learn
  "questions about article X succeed with config Y". Cross-fitting gives tighter CIs but no untouched
  test set; we keep a fixed test set for the headline and offer cross-fitting as a robustness check.
- **Risk.** Test set is ~200 queries per dataset, so per-dataset CIs will be wide. Compute the expected
  CI width from the pilot's per-query variance in week 2; if it is too wide, raise n per dataset
  (generation is cheap, see D12) rather than shrinking the test split. Hub titles still link a few queries across splits
  (`leakage_report.json` reports the rate). Retrieved-passage overlap between train and test is
  reported as a further diagnostic.

---

## 2. Harness, grid and compute

### D6. Harness: AutoRAG modules, staged and cached
- **Decision.** Use AutoRAG 0.3.27 directly: `Evaluator.start_trial` for the retrieval stage (it
  ingests BM25 and Chroma and writes every module's top-20 results), then AutoRAG's
  `SentenceTransformerReranker`, `ChatFstring` and `Vllm` modules called on the cached results
  (`harness/run_stages.py`). The 24 one-module-per-node YAMLs are generated as the formal config
  definitions, and two are run end-to-end through `Evaluator` to prove equivalence (`check_equivalence.py`).
- **Alternatives.** (a) 24 separate end-to-end AutoRAG trials. (b) One trial with all modules per node.
  (c) A LangChain / LlamaIndex pipeline (excluded by the spec).
- **Why.** (b) is wrong for us: each AutoRAG node keeps only its best module for the next node, so
  most configs would never run on most queries. (a) is correct but reloads the LLM 24 times and
  repeats BM25 retrieval (seconds per query in rank_bm25) 16 times. Staging runs retrieval once,
  reranking once per retriever, and loads each LLM once, while using AutoRAG's own code for every step.
- **Risk.** Staging could drift from AutoRAG's end-to-end semantics. Mitigation: the equivalence check
  compares retrieved ids and answers on 50 queries for two configs; agreement goes in the appendix.

**What AutoRAG does not do natively (verified in 0.3.27 source), and the workaround**

| Gap | Workaround |
|---|---|
| Node-wise greedy selection prunes the grid | One module per node; staged runner over cached stage outputs |
| `hybrid_rrf` grid-searches its RRF constant (`weight_range`, default (4, 80)) on the evaluation queries | `weight_range: (60, 60)` pins k = 60 |
| `hybrid_retrieval` needs `lexical_retrieval` and `semantic_retrieval` nodes before it | Generated hybrid configs contain all three nodes (pool 20 each) |
| `speed_threshold` in a strategy drops slow modules | Never set |
| A `prompt_maker` node with >1 module evaluates prompts with a default OpenAI generator | Exactly one prompt module |
| `vectordb: default` uses OpenAI embeddings | Named Chroma vectordb with local `BAAI/bge-base-en-v1.5` |
| No logging of reranker passes or input tokens | Reranker pairs = pool size (deterministic; logged); input tokens counted with the generator's chat template (checked against vLLM's `prompt_token_ids`) |
| Only per-module *average* time in `summary.csv` | Separate latency harness (D16) |
| No EM/F1 generation metrics (has BLEU/ROUGE/METEOR/SemScore/G-Eval/BERTScore) | Our SQuAD-style EM/F1 (`eval/metrics.py`) |
| `vllm_api` defaults to temperature 0.4 and makes one HTTP tokenize call per output token | In-process `vllm` module; temperature passed explicitly |

### D7. Retrievers
- **Decision.** BM25 (AutoRAG `bm25`, porter-stemmer tokenizer); dense `BAAI/bge-base-en-v1.5`
  (Chroma, cosine, query instruction prefix); hybrid = RRF with k = 60 over BM25 top-20 ∪ dense top-20.
- **Alternatives.** Hybrid convex combination (`hybrid_cc`) with tuned weights; e5-base-v2 or Contriever;
  RRF over top-k only.
- **Why.** RRF needs no score normalisation and no tuning (k = 60 is the original paper's default),
  which avoids tuning on test queries. bge-base is a strong, small (~110M) open embedder that AutoRAG
  supports through `llama-index-embeddings-huggingface`. Fusing a fixed pool of 20 makes hybrid top-3
  exactly the prefix of hybrid top-20, consistent with the reranker pool.
- **Risk.** Fusion choices change hybrid's rank (risk list). Sensitivity check if time permits:
  `hybrid_cc` with fixed weight 0.5 on the pilot.

### D8. top-k ∈ {3, 10}
- **Why.** 3 = cheap and precise, 10 = higher recall at ~3× input tokens; a wide gap makes the cost
  trade-off visible. 5 would add 12 configs for an intermediate point. **Risk:** coarse; the router
  cannot pick k = 5.

### D9. Reranker: cross-encoder on a fixed pool
- **Decision.** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params) via AutoRAG
  `sentence_transformer_reranker`. **Reranker on:** retrieve 20, score all 20 pairs, keep top-k.
  **Off:** retrieve top-k directly. Same pool (20) for every retriever and k.
- **Alternatives.** Rerank only the top-k (for k = 3 this merely reorders three passages); a larger
  reranker (bge-reranker-base/large); pool = 2k.
- **Why.** A fixed larger pool is how reranking is used in practice (it can bring new passages into the
  top-k) and makes rerank cost a constant 20 pairs per query, easy to count. MiniLM-L6 keeps rerank
  cost small relative to generation, which is the realistic regime.
- **Risk.** A stronger reranker might change conclusions about reranking's value; stated as a limitation.

### D10. Generators and decoding
- **Decision.** `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`, vLLM, chat template
  (`chat_fstring` → `LLM.chat`), greedy (temperature 0), `max_tokens = 32`, `max_model_len = 4096`, fp16.
- **Alternatives.** Llama-3.1-8B + Llama-3.2-3B; Mistral-7B; sampling with several seeds.
- **Why.** Same family and tokenizer, so token counts mean the same thing across generators and the
  size effect is isolated. Greedy decoding removes sampling variance from the oracle labels.
- **Hardware floor ✔.** vLLM requires NVIDIA compute capability ≥ 7.5 (official install docs): RTX 2080 Ti
  (7.5) is supported; **GTX 1080 Ti (6.1) is not**, so jobs must request 2080 Ti-class or newer GPUs.
- **Risk.** fp16 on Turing (no bf16) could cause numerical problems for some models (VERIFY in the
  pilot: look for empty or garbage outputs). vLLM greedy output can still vary slightly with batch
  composition: measured in week 1 (`greedy_reproducibility.json`).

### D11. Prompt and answer extraction
QA: system "Answer with a short phrase only"; user "Passages … Question … Short answer:". FEVER:
"Reply with exactly one word: SUPPORTS or REFUTES". Post-processing is identical for all configs: first
non-empty line, strip an "Answer:"-style prefix (`extract_answer`). No per-config prompt tuning.

### D12. Compute budget (from `scripts/budget.py`; ASSUMED throughput, replace with pilot measurements)
Assumptions: passage = 147 tokens (title + 100 words; extrapolated from the local AutoRAG smoke test, which
measured 104 Qwen tokens per ~73-word passage and mean prompts of 359 / 1,139 tokens at k=3 / k=10), prompt
overhead 80 → **521 input tokens at k=3, 1,550 at k=10**; output ~8 tokens; prefill 2,500 tok/s for 7B (TP=2 on 2×2080 Ti) and 10,000 tok/s
for 1.5B (1 GPU); ×1.3 overhead. Judge: RAGAS faithfulness + answer relevancy ≈ 3,500 input + 300
output tokens per item, 7–9B judge on 2 GPUs at 2,500 tok/s.

| Datasets | Queries | LLM calls (24 configs) | Gen. input tokens | **Gen. GPU-h** | Judge on ALL items, GPU-h | Judge on all items, API $ (illustrative†) |
|---|---|---|---|---|---|---|
| NQ + HotpotQA | 2,000 | 48,000 | 49.7M | **8.1** | 48.5 | 34 (cheap tier) / 564 (strong tier) |
| + TriviaQA | 3,000 | 72,000 | 74.6M | **12.1** | 72.8 | 51 / 846 |
| + FEVER | 4,000 | 96,000 | 99.4M | **16.2** | 97.1 | 68 / 1,128 |

† Illustrative prices only: "cheap tier" $0.15 / $0.60 per 1M input/output tokens, "strong tier"
$2.50 / $10. **VERIFY against current price sheets; never quote these as real costs.**
If measured throughput is half the assumption, generation roughly doubles (about 16 / 24 / 32 GPU-h).

Offline (once): sub-corpus embedding of 1M passages with bge-base ≈ 0.5 GPU-h (ASSUMED); optional
full-corpus dense mining ≈ 21× that (~10 GPU-h); BM25 ingestion is CPU-only.
Reranker: 3 retrievers × 20 pairs × 4,000 queries = 240,000 pairs once; minutes with MiniLM-L6.

**What the arithmetic implies.**
1. The full grid is affordable: about 16 GPU-h for 4 datasets, ~32 GPU-h with reruns. Under the cs4406 QoS
   (1 GPU, 6 h per job) that is roughly 4–6 sequential 6-hour jobs.
2. **Judging every output costs ~6× the whole generation budget.** So the LLM judge cannot be the
   oracle label; EM/F1 are (free and deterministic). RAGAS runs only on a validated subsample:
   100 test queries/dataset × 24 configs = 9,600 items ≈ 9.7 GPU-h (200/dataset ≈ 19.4 GPU-h).
3. Each extra dataset costs ~4.0 generation GPU-h but ~24 judge GPU-h if fully judged. This is why
   dataset count is limited by the judge, not by generation.

---

## 3. Metrics and statistics

### D13. Primary quality metric (pre-committed)
- **EM and token-F1** on short gold answers, SQuAD normalisation, max over gold aliases.
  **Router target and oracle label: per-query F1** (QA) / **accuracy** (FEVER).
- **Why F1 as target, not EM.** EM is binary and strict (a correct "President Obama" vs gold "Obama"
  scores 0), which inflates ties and label noise in the oracle. F1 is graded. EM is still reported
  everywhere, and the 150-item human study measures EM's false-negative rate.
- Lenient "contains gold" is a diagnostic only, never a headline number.

### D14. Normalisation across datasets/metrics
- **Decision.** Keep every per-query score in its native [0, 1] scale (F1 for QA, accuracy for FEVER).
  No per-config or per-query rescaling. Report per dataset always; pooled numbers both micro
  (per query) and macro (equal weight per dataset).
- **Alternatives and pitfalls.**
  - *Per-dataset z-score across configs:* depends on which configs are in the grid (adding a config
    changes every score); amplifies noise where configs barely differ; not interpretable as quality.
  - *Per-query min-max across configs:* collapses to 0/1, undefined when all configs tie (very common:
    all succeed or all fail), and throws away the absolute level the router needs to trade off against cost.
  - *Per-dataset min-max:* same set-dependence as z-scores.
- **Remaining pitfall.** Accuracy on 2-way FEVER has a 0.5 chance floor while QA F1 has ~0, so pooled
  means mix scales. Mitigations: macro reporting, headline results with and without FEVER, and a
  chance-corrected FEVER score (2·acc − 1) as a sensitivity check.

### D15. Secondary metrics: RAGAS after validation
- RAGAS faithfulness and answer relevancy (`eval/ragas_eval.py`), judge served by vLLM, judge model
  **not** from the Qwen family (self-preference risk).
- Validation first: 150 hand-labelled dev items (docs/05). Report judge-vs-human Spearman/Pearson and
  Cohen's κ (judge binarised at 0.5) with bootstrap CIs, plus inter-annotator κ.
- **Pre-registered rule:** use RAGAS at scale only if κ ≥ 0.6 (team to confirm) and the Spearman CI
  lower bound > 0.4; otherwise report it only as an unreliable diagnostic.
- Short-answer caveat: with one-phrase answers, faithfulness reduces to one entailment check and
  relevancy is near-saturated. Judge noise can exceed differences between configs. That is why RAGAS
  never drives routing.

### D16. Statistics
- **Paired bootstrap** (`stats/bootstrap.py`): resample queries with replacement, 10,000 resamples,
  stratified by dataset (each resample keeps the test set's dataset mix); 95% percentile CI;
  two-sided null-centred p-value. Used for every config-vs-config and router-vs-baseline claim.
- **Seeds.** Decoding is greedy, so there is no sampling seed for generation. Its residual
  nondeterminism is measured. Router training uses 5 seeds (0–4); per-query outcomes are averaged over
  seeds before the query bootstrap, and seed SD is reported separately.
- **Multiple comparisons.** Holm–Bonferroni within each family (e.g. all 23 configs vs the best; all
  router variants vs best static).
- **Per-question-type "best config" claims** (`best_config_stability`): bootstrap frequency of each
  config being best, and a paired CI for (best − runner-up). The claim is made only if that CI excludes 0.
- **Oracle null** (`oracle.oracle_null`): permute each config's outcomes across queries within dataset
  (200 permutations) and recompute the oracle. "max over 24 noisy outcomes" inflates the oracle; only
  real − null is potentially predictable headroom.

---

## 4. Cost model

### D17. Countable marginal cost with generator-specific token weights
- **Decision.** `cost = a_in[g]·input_tokens + a_out[g]·output_tokens + b·llm_calls + c·rerank_passes + d·index_queries`
  (`cost.py`, `configs/cost_weights.yaml`). Units: GPU-ms equivalents, calibrated from the week-1 pilot
  (`stage_timings.jsonl`): a_in[g] = GPUs × 1000 / prefill tok/s, etc.
- **Deviation from the spec's single `a`, and why.** One weight for all tokens would make a 7B token
  cost the same as a 1.5B token, so the generator choice would be free. `a` is therefore indexed by
  generator, and input/output tokens are separated (decode is far costlier per token). Setting
  `a_in = a_out = a` for both generators recovers the spec's formula as one of the sensitivity sets.
- **Counts.** input tokens from the chat-templated prompt; output tokens from vLLM; llm_calls = 1;
  rerank_passes = 20 if reranker on; index_queries = 1 (BM25), 1 (dense, incl. query encoding),
  2 (hybrid).
- **Offline cost** (once per corpus; `scripts/record_offline_cost.py`): embedding GPU-ms and BM25
  build time (timed on a 50k slice and extrapolated, labelled as such), Chroma and BM25 index sizes.
  **Reported separately.** Folded in only through an explicit amortisation factor (queries served per
  build ∈ {1e3, 1e4, 1e5, 1e6}, default 1e5) in table T9. BM25-only configs are not charged for the
  dense index.
- **Weight sensitivity** (T8): named alternative weight sets (tokens-only, params-scaled, API-price-like)
  plus 500 log-uniform draws (each weight × [1/4, 4]); report Kendall τ of static config cost rankings,
  stability of the static Pareto frontier, and re-run the main routers with each named set.
- **Risk.** Weights are a modelling choice; the sensitivity analysis decides how much any conclusion
  depends on them.

### D18. Latency (secondary)
p50/p95 of end-to-end per-query latency (batch size 1) over 3 repeats of 200 test queries, config order
randomised per repeat, warm-up discarded, both generators in **one SLURM job on one node**
(`eval/latency.py`, `slurm/06_latency.sbatch`). Rows carry session id, host and GPU; the table code
refuses to mix sessions. Caveat: AutoRAG BM25 is pure-Python rank_bm25, so its latency is not
representative of production BM25 (Lucene).

---

## 5. The router

### D19. Oracle table
`harness/collect.py` → one row per (query, config): quality (F1/acc), EM, cost components, scalar
cost, split, dataset, question type. Oracle policies (`oracle.py`): **oracle_best** (max quality,
ties → cheapest), **oracle_cheapest_ok** (cheapest config with EM-style success), **oracle_λ**
(argmax q − λc on true labels; the oracle Pareto curve). Label diagnostics: share of queries where all
configs fail / all succeed / mixed, number of distinct oracle-best configs, top-label share.

### D20. Formulation: pooled quality regression, then argmax q̂ − λĉ (recommended)
- **Decision.** One model predicts q̂(query, config) ∈ [0, 1] from [query features, data features,
  config encoding (+ interactions for linear/MLP models)]. ĉ(query, config) = train mean cost of that
  config within the query's dataset (cost is near-deterministic given config and dataset; its
  within-config coefficient of variation is reported in T2 to justify this). Route by
  argmax_c q̂ − λ·ĉ/scale, where scale is one fixed constant. Sweeping λ gives the router's Pareto curve
  **without retraining**.
- **Alternative: classify the best (or cheapest-sufficient) config** (24-way, or per-config success
  classifiers).
- **Why the regression formulation.** (1) Ties are the norm (all succeed / all fail), so "the best
  config" is ill-defined for many queries and tie-breaking by cost turns the label into "cheapest
  config"; regression keeps all 24 outcomes per query. (2) Label imbalance: one or two configs dominate
  argmax labels, so a classifier collapses to the majority class. (3) One model serves every λ and
  every cost-weight set; a classifier needs relabelling and retraining per λ. (4) Statistical strength
  is shared across configs.
- **Risk.** The model may learn only config main effects, which reproduces the best static config.
  Detect it by comparing with `D_only` and static baselines; LR and MLP get explicit query × config interactions.
- **Models:** fractional logistic regression (soft-label cross-entropy via weighted duplication),
  LightGBM with cross-entropy objective, small MLP (128-64); 5 seeds each.

### D21. Feature groups
| Group | Features | Available |
|---|---|---|
| 1. Query | length (tokens, chars); wh-word one-hots; auxiliary-verb start (yes/no); capitalised-span entity count; years/numbers/quotes; comparison cues (both/than/older/first…, "X or Y"); bridge cues (relative pronouns, "of the" chains, possessives); PCA-32 of the bge question embedding (fit on train); learned multi-hop score (out-of-fold LR on train) | before retrieval |
| 2. Data | dataset one-hots; dataset attributes (multi-hop, verification task, trivia style) | before retrieval |
| 3. Config | retriever one-hot, top-k, reranker, large generator | — |
| 4. Metric | observed quality and cost of the pair: **training target only, never an input** | — |
| Router B probe | per retriever (BM25, dense top-20): s1, s2, s10; relative gaps s1−s2 and s1−s10; relative std; score entropy (shift-normalised); query content-word overlap with top-1 / best of top-3; distinct titles in top-10; title-in-question; cross: Jaccard@3/@10 and top-1 agreement of BM25 vs dense | after a 2-query probe |

### D22. Router B cost accounting
Probe = BM25 top-20 + dense top-20 = 2 index queries (the dense one includes the question embedding).
**With reuse (default):** the chosen config's own retrieval is served from the probe (BM25, dense, and
hybrid = RRF of the two pools are all derivable), so its own index queries are refunded.
**Conservative (`--no-reuse`):** probe cost is added on top. Both reported. Router A pays one
question-embedding pass unless it picks a dense/hybrid config. No reranker scores or LLM calls are used
as signals; they would not be cheap.

### D23. Baselines (all fit on train)
Best static (argmax train quality); best static under a cost budget, i.e. the static λ-frontier;
cheapest config; uniform random router (exact per-query expectation); **per-dataset best static**
(uses dataset identity: tests how much "routing" is just knowing the dataset); **Adaptive-RAG-style**
(LR complexity classifier on question text → {single, multi, verification} → best static per class,
with the classifier's cost charged as Router A); oracle (best, λ-curve). Reference only: the best static
chosen on test (optimistic).

### D24. Evaluation protocol
- λ grid: 0 plus 60 log-spaced values in [1e-3, 5]. λ is selected **on dev** for (a) **iso-cost**: max dev
  quality with dev cost ≤ best static's dev cost; (b) **iso-quality**: min dev cost with dev quality ≥
  best static's. Evaluated once on test.
- Headline numbers per router: ΔQ vs best static at iso-cost; % cost saving at iso-quality (with ΔQ CI
  as non-inferiority evidence); **gap closed** = (Q_router − Q_static) / (Q_oracle@same cost − Q_static).
- Breakdowns: per dataset, per hops type, per HotpotQA type (bridge / comparison), each with paired bootstrap.
- Ablations: feature sets {A_noDS, A, A_noEmb, B, B_noDS, D_only, R_only} × models {LR, GBDT, MLP} ×
  5 seeds; λ sweep; reuse vs no-reuse; cost-weight sets.
- Report where the router does not help, with CIs (section "Where routing does not help").
