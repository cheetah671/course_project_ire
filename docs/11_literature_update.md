# Literature update (7 Nov 2026): reference and citation check requested by the TA

## How the search was done
The TA asked us to stop relying on keyword search and to check who cites the core papers. On 7 Nov we pulled
every citing paper of six seed papers from OpenAlex (Semantic Scholar's API was rate-limited):

| Seed | Citing papers found |
|---|---|
| Wang et al. 2024, Searching for Best Practices in RAG | 243 |
| AutoRAG (arXiv:2410.20878) | 44 |
| AutoRAG-HP (Findings of EMNLP 2024) | 25 |
| Orbach et al. 2025, HPO for RAG (arXiv:2505.03452) | 4 |
| Adaptive-RAG (NAACL 2024) | 558 |
| METIS (SOSP 2025) | 35 |

Script: `scripts/lit_openalex_citations.py OUTDIR` (no API key needed; page-number paging).
The 909 titles were filtered by topic keywords (routing, per-query, configuration, cost, efficiency, Pareto,
budget, chunking, hyper-parameters, complexity), which left 240 candidates. These were screened by hand.
The abstracts of the papers below were then read (arXiv or OpenAlex).


## Papers that change how we position the project

| Paper | What it does | Consequence for us |
|---|---|---|
| **BRANE**: Pan, Arabzadeh, Jacob, Kazhamiaka, Choukse, Zaharia, "Natural Language Query to Configuration for Retrieval Agents", arXiv:2605.27361 (May 2026) | Per-query choice of a full retrieval-pipeline configuration (LLM, retriever, number of documents, hops, synthesis) from a catalogue. An LLM turns the query into workload-specific characteristics; a light per-configuration predictor estimates correctness; the configuration maximising predicted correctness minus a cost penalty is chosen, with a tunable trade-off and no retraining. Matches the best fixed configuration's accuracy at up to 89% lower cost on MuSiQue, BrowseComp-Plus and FinanceBench. | **Closest prior work; our routing formulation (q̂ − λĉ, one model for every λ) is essentially theirs.** We do not claim the formulation. What differs: (1) our features are cheap (surface cues, a small embedding, retrieval-score signals); BRANE calls an LLM per query to build its features, a cost we would have to charge; (2) Router B's post-retrieval signals; (3) three tasks with different metrics, and global vs per-task routers; (4) countable marginal cost including router overhead, kept apart from amortised offline cost; (5) a leakage-controlled grouped split; (6) we report whether learned routers beat the best static config at all (see RCT-MARS), not only Pareto improvements. |
| **RCT-MARS**: Srivastava, "when per-query retrieval routing fails, and what it takes to succeed", J. Intelligent Information Systems (2026) | Names the **routing paradox**: an oracle choosing the best of 5 retrieval paradigms per query beats every fixed paradigm (0.599 vs 0.465 nDCG@10), yet the best hard router falls 7.8 pp *below* the best fixed paradigm. Soft routing (per-query fusion weights) recovers +16% of the oracle gap, matching unsupervised RRF. | Our pilot showed the same pattern (oracle +0.196 F1 over best static; no router beat it). Our full run is a test of the paradox **end to end** (retrieval + rerank + generator) **with cost counted**. A null result is a replication, not a failure. Their soft-routing escape does not apply directly to generator choice; we note it as future work. |
| **Arabzadeh, Drozdov, Bendersky, Zaharia**, "Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines", SIGIR 2026 | Query performance prediction (QPP) used to pick among query reformulations before running the pipeline. Ranking quality and answer quality diverge (a "utility gap"); cheap pre-retrieval predictors often match or beat post-retrieval ones. | Router B's probe signals (score gaps, spread, BM25/dense agreement) **are post-retrieval QPP predictors** (WIG/NQC-like). We now frame Router A vs B as "pre- vs post-retrieval QPP for configuration choice", and their result is a prior for H2 (B > A) being small. Classic QPP to cite: Cronen-Townsend, Zhou & Croft, "Predicting Query Performance", SIGIR 2002 (Clarity); Zhou & Croft, "Query Performance Prediction in Web Search Environments", SIGIR 2007 (WIG); Shtok, Kurland, Carmel, Raiber & Markovits, "Predicting Query Performance by Query-Drift Estimation", ACM TOIS 2012 (NQC). |
| **RAGRouter-Bench**: Wang, Zhu, Lin, Xue, Guo, Zhang, arXiv:2602.00296 (Jan 2026) | Benchmark for routing between RAG *paradigms* using query type and corpus indicators; reports quality and resource use; no single paradigm wins; adaptive routing improves the effectiveness–efficiency trade-off. | Related benchmark at the paradigm level (naive / graph / iterative RAG). We route among configurations of one paradigm, on fixed corpora, and add a cost-counted oracle ceiling. |
| **Qian et al.**, "When Should Active RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost", KDD 2026 workshop, arXiv:2607.24010 | Evaluation protocol for retrieve/not-retrieve routers: frontiers, realised budgets, threshold-transfer error, cost decomposition. Simple uncertainty or retrieval-score baselines often rival learned utility routers. | Supports our protocol (λ chosen on dev, evaluated once on test; frontier reporting) and our simple baselines. We report realised (test) cost, not the dev target. |
| **Su, Healey, Nakov, Cardie**, "Fast or Better? Balancing Accuracy and Cost in RAG with Flexible User Control", arXiv:2502.12145 (2025) | Two classifiers (accuracy-first, efficiency-first) mixed by a user parameter α over no / single / iterative retrieval. | Same idea of a user-set trade-off; our λ plays the role of α over a larger configuration space. |
| **Zhang et al.**, "RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models", NeurIPS 2025, arXiv:2505.23052 | Routes each query to one of several LLMs under RAG; retrieved documents change which LLM is best, so routers must see them. | Matches our generator arm (1.5B vs 7B) and motivates post-retrieval features. |
| **Barker et al.**, "Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems", arXiv:2502.18635 (2025) | Bayesian multi-objective optimisation of cost, latency, safety and alignment over whole RAG systems; optimal configurations do not transfer across tasks and objectives. | Static (per-workload) Pareto search; complements our per-query view. Their non-transfer finding motivates per-task routers. |
| **HAMMER**: Zhou, Wang, Fang, Proc. ACM Manag. Data (SIGMOD) 2026 | MCTS with hierarchical memory to tune RAG pipelines; up to 9× less tuning time and tokens. | Another static tuner; cite next to AutoRAG-HP and Orbach et al. |
| **Varangot-Reille et al.**, "Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in LLM-Based Systems", JAIR 2026 | Survey of pre- vs post-generation routing for cost. | Background for routing; our routers are pre-generation. |

## Chunking versus cost (fills the TODO in docs/01, §2)
| Paper | Note |
|---|---|
| Śmigielski et al., "Chunking Methods on Retrieval-Augmented Generation – Effectiveness Evaluation Against Computational Cost and Limitations", Procedia Computer Science (2026) | The paper our original proposal cited. Compares many chunking methods with computational cost. Chunking stays fixed in our design (D3), so it is background only. |
| Galimzyanov, Kolomyttseva, Bogomolov, "Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets", arXiv:2510.20609 (2025) | Chunking, scoring and splitting under compute budgets for code; retrieval latency varies up to 200× across configurations; BM25 is the best quality–latency trade-off for code-to-code. |

## Screened and judged not needed
Titles that matched keywords but are about other problems (agentic search efficiency, KV-cache reuse, domain
RAG applications, multimodal routing, adaptive listwise reranking) were not cited. If the report needs more
on "when to retrieve", Adaptive-RAG and the budget-aware Active RAG paper above cover it.

## What changes in the report
1. Related work gets the table above (compressed). BRANE and RCT-MARS go in the introduction.
2. Contributions are restated: not the routing formulation (BRANE has it), but (a) an end-to-end, cost-counted
   test of the routing paradox across three tasks, (b) pre- vs post-retrieval (QPP-style) signals at their real
   cost, (c) global vs per-task routers across metrics, (d) leakage-controlled evaluation.
