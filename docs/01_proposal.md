# Deliverable 1: Proposal / report skeleton

> Fill-in convention: `[[...]]` marks a result we do not have yet. Never replace one with a guess.
> Citations marked VERIFY must be checked against the paper itself before submission.

---

## Title
**Which RAG Configuration for Which Query? Cost-Aware Per-Query Configuration Routing**

## Abstract (two versions; keep the one the results support)

**Version W (router wins).** Retrieval-augmented generation pipelines expose many configuration
choices, and most evaluations report answer quality only. We run all 24 configurations of a
retriever × top-k × reranker × generator grid on every query of [[N]] questions from NQ-open,
HotpotQA [[, TriviaQA, FEVER]], turning the sweep into per-query oracle labels with countable costs.
The per-query oracle exceeds the best static configuration by [[Δ]] F1 at equal cost. A router using
only the question recovers [[x]]% of that gap; adding cheap post-retrieval signals recovers [[y]]%,
saving [[z]]% marginal cost at matched quality (paired bootstrap, 95% CI [[...]]).

**Version S (small gain).** ... The per-query oracle exceeds the best static configuration by only
[[Δ]] F1, and a permutation test shows that [[most]] of this headroom is consistent with outcome noise
rather than predictable query–configuration structure. Routers recover [[x]]% of the gap, which is not
significant ([[CI]]). This explains why static best-practice recipes work well on these benchmarks,
and identifies [[where, e.g. multi-hop bridge questions]] as the only region where routing pays.

## 1. Introduction and motivation
- RAG has many interacting configuration choices (chunking, retrieval, reranking, generation).
  Best-practice studies (Wang et al., 2024) and AutoML-style tools (AutoRAG; AutoRAG-HP) search for
  one good configuration per dataset.
- Two gaps:
  1. **Cost is under-reported, especially generation-side cost.** Systems papers focus on latency
     and serving; quality papers ignore cost. A team needs to know whether the extra quality of
     reranking, larger k or a larger generator is worth its tokens and calls.
  2. **One configuration for all queries may waste cost.** An easy single-hop question may be solved
     by BM25, k=3 and a 1.5B model; a bridge question may need hybrid retrieval, k=10 and a 7B model.
- Idea: an exhaustive sweep usually yields only a table. We use it differently: running every config
  on every query gives **oracle labels** (which configs succeed, at what cost), and the question
  becomes whether the best config is *predictable per query*.
- Contributions (phrase according to outcome):
  1. A fully crossed, cost-instrumented 24-config grid with per-query oracle labels, released as a table.
  2. The oracle ceiling vs best static config on the cost–quality Pareto frontier, with a
     permutation test separating predictable headroom from noise.
  3. Routers from question-only features and from cheap post-retrieval signals, with the signal cost
     charged to the router, compared with static, per-dataset and Adaptive-RAG-style baselines.
  4. A countable cost model (offline vs marginal, amortisation, weight sensitivity).

## 2. Related work (positioning; we claim no novelty over these)

| Work | What it does | How we differ / what we reuse |
|---|---|---|
| Wang et al., "Searching for Best Practices in Retrieval-Augmented Generation", EMNLP 2024 main (2024.emnlp-main.981; arXiv:2407.01219) ✔ | Full-pipeline ablation; recommends a "Balanced Efficiency Practice" (term from the spec: confirm wording in the paper); reports latency | They pick one recipe per setting. We ask whether the recipe should vary per query, and use countable cost rather than wall time alone. |
| AutoRAG (Kim, Kim, Han, Eibich; arXiv:2410.20878, 2024; arXiv only, no venue listed) ✔ | Automated module search, node-by-node; per-query speed threshold; open source | **Our harness.** Its node-wise greedy selection prunes the grid; we disable that (one module per node, staged caching) to obtain all 24 configs on all queries. |
| AutoRAG-HP (Fu et al., Findings of EMNLP 2024, pp. 3875–3891; arXiv:2406.19251) ✔ | Online hyper-parameter tuning as a multi-armed bandit (plus hierarchical MAB); reaches Recall@5 ≈ 0.8 with ~20% of the LLM API calls grid search needs | Makes exhaustive grids the known-wasteful way to *find a config*. We run the grid for another reason: oracle labels need every cell. We report its cost explicitly. |
| Orbach et al., "An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation" (arXiv:2505.03452; AAAI 2026 Workshop on New Frontiers in IR) ✔ | Benchmarks 5 HPO algorithms on 5 datasets; greedy or random search is efficient; for greedy methods, choosing the model first beats pipeline order | Same: HPO finds one config per dataset; we study per-query variation. Their "model first" finding is a useful prior for our generator feature. |
| Adaptive-RAG (Jeong, Baek, Cho, Hwang, Park; NAACL 2024; arXiv:2403.14403) ✔ | Classifier of query complexity → no-retrieval / single-step / multi-step strategy | Closest per-query baseline. It uses question text only; we add post-retrieval signals and a cost objective, and re-implement an Adaptive-RAG-*style* complexity → config baseline. |
| METIS (Ray, Pan, Gu, Du, Feng, Ananthanarayanan, Netravali, Jiang; SOSP 2025; arXiv:2412.10543, first posted as "RAGServe") ✔ | Serving system that jointly schedules queries and adapts per-query RAG configuration (e.g. number of retrieved chunks, synthesis method); reports 1.64–2.54× lower generation latency without quality loss | Discussed conceptually: it optimises serving latency under load; we do not reproduce a serving system (§5.4). |
| RAGAS (Es, James, Espinosa-Anke, Schockaert; EACL 2024 System Demonstrations, 2024.eacl-demo.16, pp. 150–158) ✔ | LLM-judge metrics (faithfulness, answer relevancy) | Secondary metric only, after human validation. |
| LegalBench-RAG (Pipitone, Houir Alami; arXiv:2408.10343, 2024; no venue listed) ✔ and other domain RAG benchmarks | Domain-specific retrieval benchmarks | Out of scope; listed as future work for domain transfer. |
| Chunking-vs-cost studies | [[TODO: find via reference lists of the first four papers; do not cite unread]] | We fix chunking (§design D3). |

**TODO (open item 6):** pull the reference lists of the first four papers and check "cited by" for each.

## 3. Research questions and hypotheses
- **RQ1 (ceiling).** How much better is the per-query oracle than the best static config, at equal
  marginal cost, and how much of that gap survives a permutation null (outcome noise)?
- **RQ2 (Router A).** What fraction of the predictable gap does a router recover from the question
  text alone (before any retrieval)?
- **RQ3 (Router B).** What does it recover once cheap post-retrieval signals are added, *after*
  charging the cost of computing them?
- **RQ4 (where).** For which datasets and question types (single vs multi-hop; bridge vs comparison)
  does routing help, and where does it not?
- **RQ5 (cost model).** Are conclusions stable under cost-weight changes and amortisation factors?

Hypotheses (pre-registered direction, not values): H1 oracle > static, beyond the null. H2 Router B ≥
Router A at matched cost. H3 gains concentrate in multi-hop questions. H4 a large share of routing
value comes from dataset identity alone (tested by the per-dataset static baseline and the `D_only` ablation).

## 4. Method (summary; full detail and decision records in docs/02_experiment_design.md)
4.1 Data: NQ-open and HotpotQA (+ TriviaQA, optionally FEVER), ~1,000 queries each, grouped
train/dev/test split by query with leakage checks.
4.2 Corpus and chunking: DPR 100-word Wikipedia passages (fixed chunking); ~1M-passage sub-corpus
with full-corpus BM25 hard negatives.
4.3 Grid: {BM25, dense bge-base, hybrid RRF k=60} × k∈{3,10} × reranker {off, MiniLM-L6 cross-encoder
on a fixed pool of 20} × {Qwen2.5-7B, Qwen2.5-1.5B}, greedy decoding = 24 configs.
4.4 Harness: AutoRAG modules, staged and cached (retrieve once, rerank once, generate per generator).
4.5 Metrics: EM and token-F1 (primary, pre-committed); accuracy for FEVER; RAGAS on a validated subsample.
4.6 Cost: countable marginal cost (tokens, calls, rerank pairs, index queries) with generator-specific
token weights; offline cost reported separately with amortisation.
4.7 Oracle table and routers: pooled quality model over (query, config) pairs; route by
argmax q̂ − λĉ; Router A (question only) and Router B (+ probe signals, probe cost charged).

## 5. Evaluation plan
5.1 Held-out grouped test split; λ chosen on dev at (a) iso-cost and (b) iso-quality with the best static config.
5.2 Cost–quality Pareto frontier: 24 static configs, static frontier, per-dataset static,
Adaptive-RAG-style, Router A, Router B, oracle.
5.3 Statistics: paired bootstrap (10k resamples, stratified by dataset) for every comparison;
Holm correction within families; per-question-type best-config claims only when the margin CI excludes 0.
5.4 METIS (conceptual): it adapts configuration online under serving load with an LLM profiler; our
setting is offline and countable-cost. We discuss which of our signals a METIS-like system could use
and what our oracle analysis says about the value of its per-query adaptation in principle.
5.5 Ablations: feature groups, router model (LR / GBDT / MLP), λ, cost weights, probe reuse accounting.
5.6 Honest reporting: a dedicated subsection "Where routing does not help" with CIs.

## 6. Risks (top five; full register in docs/08_risks.md)
Oracle headroom is mostly noise · label imbalance (one config dominates) · EM strictness ·
judge noise · Ada queue/time limits.

## 7. Timeline
8 weeks, 3 people (docs/07_plan_and_team_split.md). Plumbing is week 1; weeks 2–7 go to oracle
labelling, routing and evaluation.

## References
Core works 1–8: title, authors, venue and IDs checked on arXiv / ACL Anthology on 2026-10-27 (✔).
Still to do: read each paper before citing a claim from it, pull their reference lists, check "cited by".
1. ✔ Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang,
   Shizheng Li, Qi Qian, Ruicheng Yin, Changze Lv, Xiaoqing Zheng, Xuanjing Huang. Searching for Best Practices
   in Retrieval-Augmented Generation. EMNLP 2024 (main), pp. 17716–17736. doi:10.18653/v1/2024.emnlp-main.981. arXiv:2407.01219.
2. ✔ Dongkyu Kim, Byoungwook Kim, Donggeon Han, Matouš Eibich. AutoRAG: Automated Framework for optimization of
   Retrieval Augmented Generation Pipeline. arXiv:2410.20878, 2024.
3. ✔ Jia Fu, Xiaoting Qin, et al. AutoRAG-HP: Automatic Online Hyper-Parameter Tuning for Retrieval-Augmented
   Generation. Findings of EMNLP 2024, pp. 3875–3891. arXiv:2406.19251. (Full author list: copy from the Anthology page.)
4. ✔ Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni, Yoav Kantor, Gal Bloch, Omri Levy,
   Hadas Abraham, Nitzan Barzilay, Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo.
   An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation. AAAI 2026 Workshop on
   New Frontiers in Information Retrieval. arXiv:2505.03452.
5. ✔ Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park. Adaptive-RAG: Learning to Adapt
   Retrieval-Augmented Large Language Models through Question Complexity. NAACL 2024. arXiv:2403.14403.
6. ✔ Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh Ananthanarayanan, Ravi Netravali,
   Junchen Jiang. METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation. SOSP 2025.
   doi:10.1145/3731569.3764855. arXiv:2412.10543 (v1 titled "RAGServe").
7. ✔ Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. RAGAs: Automated Evaluation of Retrieval
   Augmented Generation. EACL 2024 System Demonstrations, pp. 150–158. arXiv:2309.15217.
8. ✔ Nicholas Pipitone, Ghita Houir Alami. LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the
   Legal Domain. arXiv:2408.10343, 2024.
Datasets/tools we use (cite their original papers): Natural Questions (Kwiatkowski et al., TACL 2019);
NQ-open (Lee et al., ACL 2019); HotpotQA (Yang et al., EMNLP 2018); TriviaQA (Joshi et al., ACL 2017);
FEVER (Thorne et al., NAACL 2018); KILT (Petroni et al., NAACL 2021) if FEVER-KILT is used; DPR
(Karpukhin et al., EMNLP 2020); Reciprocal Rank Fusion (Cormack et al., SIGIR 2009); BGE / C-Pack
(Xiao et al., 2023); Sentence-BERT (Reimers & Gurevych, EMNLP 2019) and MS MARCO (Nguyen et al., 2016)
for the cross-encoder; Qwen2.5 technical report (2024); vLLM / PagedAttention (Kwon et al., SOSP 2023);
Pyserini (Lin et al., SIGIR 2021); Cohen's kappa (Cohen, 1960); bootstrap significance testing for NLP
(Koehn, EMNLP 2004; Berg-Kirkpatrick et al., EMNLP 2012). These dataset/tool references are not yet
checked: VERIFY each before submission.
