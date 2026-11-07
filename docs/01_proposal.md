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
[[Δ]] F1, but routers trained on held-out questions recover only [[x]]% of that gap, which is not
significant ([[CI]]): most of the per-query headroom is not predictable from the signals we tested. This explains why static best-practice recipes work well on these benchmarks,
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
     label statistics (all-fail / all-succeed / mixed) showing where routing has room at all.
  3. Routers from question-only features and from cheap post-retrieval signals, with the signal cost
     charged to the router, compared with static, per-dataset and Adaptive-RAG-style baselines.
  4. A countable cost model (offline vs marginal, amortisation, weight sensitivity).
- Positioning after the 7 Nov literature check (docs/11): per-query configuration routing with a cost-penalised
  predictor already exists (BRANE, 2026), and RCT-MARS (2026) reports a "routing paradox" (large oracle headroom
  that learned routers do not recover). Our contribution is therefore an end-to-end, cost-counted test of that
  paradox across three tasks, comparing pre- and post-retrieval (QPP-style) signals at their real cost.

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
| **BRANE** (Pan, Arabzadeh, Jacob, Kazhamiaka, Choukse, Zaharia; "Natural Language Query to Configuration for Retrieval Agents", arXiv:2605.27361, 2026) ✔ abstract read | Per-query choice of a whole retrieval-pipeline configuration from a catalogue; an LLM extracts query characteristics, per-configuration correctness predictors, argmax of predicted correctness minus a cost penalty; matches the best fixed configuration at up to 89% lower cost | **Closest prior work: the routing formulation is theirs, not ours.** We differ in cheap features (no LLM call per query), post-retrieval (QPP-style) signals, three tasks with different metrics and global vs per-task routers, countable marginal cost with router overhead charged, leakage-controlled splits, and an explicit test of whether routers beat the best static config at all. |
| RCT-MARS (Srivastava; J. Intelligent Information Systems, 2026) ✔ abstract read | Names the "routing paradox": the per-query oracle beats every fixed retrieval paradigm, yet learned hard routers fall below the best fixed one; soft fusion recovers part of the gap | Our study tests the paradox end to end (retrieval, rerank, generator) with cost counted. A null result is a replication of the paradox. |
| Arabzadeh, Drozdov, Bendersky, Zaharia, "Can QPP Choose the Right Query Variant?", SIGIR 2026 ✔ abstract read; classic QPP: Cronen-Townsend et al. SIGIR 2002 (Clarity), Zhou & Croft SIGIR 2007 (WIG), Shtok et al. ACM TOIS 2012 (NQC) | Query performance prediction for choosing among RAG query variants; cheap pre-retrieval predictors often match post-retrieval ones | Router B's probe signals are post-retrieval QPP predictors (score gaps, spread, agreement). Router A vs B is pre- vs post-retrieval QPP for configuration choice. |
| RAGRouter-Bench (Wang et al., arXiv:2602.00296, 2026); RAGRouter (Zhang et al., NeurIPS 2025, arXiv:2505.23052); Su et al., "Fast or Better?" (arXiv:2502.12145, 2025) ✔ abstracts read | Routing between RAG paradigms (query + corpus features); routing among retrieval-augmented LLMs (documents change which LLM is best); a user-set accuracy/cost knob over retrieval strategies | Same family of per-query routing. We route among configurations of one paradigm, include the generator in the choice, and charge the router's own cost. |
| Static multi-objective tuners: Barker et al., "Faster, Cheaper, Better" (arXiv:2502.18635, 2025); HAMMER (Zhou, Wang, Fang; Proc. ACM Manag. Data, 2026) ✔ abstracts read | Bayesian multi-objective and MCTS tuning of whole RAG pipelines (cost, latency, quality); optimal configs do not transfer across tasks | One configuration per workload; motivates our per-task routers. |
| Chunking vs cost: Śmigielski et al., Procedia Computer Science (2026); Galimzyanov et al., "Practical Code RAG at Scale" (arXiv:2510.20609, 2025) ✔ abstracts read | Chunking methods compared with computational cost; retrieval design under compute budgets (code) | Chunking is fixed in our design (D3); background only. |

Reference and citation check (open item 6): done on 7 Nov 2026 with OpenAlex over 909 papers citing Wang et al.,
AutoRAG, AutoRAG-HP, Orbach et al., Adaptive-RAG and METIS. Method and screening: docs/11_literature_update.md.

## 3. Research questions and hypotheses
- **RQ1 (ceiling).** How much better is the per-query oracle than the best static config, at equal
  marginal cost, and on what share of questions do configurations disagree at all? (Whether that gap is
  *predictable* is answered out of sample by RQ2–RQ3; an oracle-level null cannot answer it: docs/02 D16.)
- **RQ2 (Router A).** What fraction of the predictable gap does a router recover from the question
  text alone (before any retrieval)?
- **RQ3 (Router B).** What does it recover once cheap post-retrieval signals are added, *after*
  charging the cost of computing them?
- **RQ4 (where).** For which datasets and question types (single vs multi-hop; bridge vs comparison)
  does routing help, and where does it not?
- **RQ5 (cost model).** Are conclusions stable under cost-weight changes and amortisation factors?

Hypotheses (pre-registered direction, not values): H1 oracle > static (CI excludes 0). H2 Router B ≥
Router A at matched cost. H3 gains concentrate in multi-hop questions. H4 a large share of routing
value comes from dataset identity alone (tested by the per-dataset static baseline and the `D_only` ablation).

## 4. Method (summary; full detail and decision records in docs/02_experiment_design.md)
4.1 Data: three tasks over one Wikipedia index: open-domain QA (NQ-open, HotpotQA, TriviaQA), fact
verification (FEVER, KILT) and slot filling (zsRE, KILT); ~1,000 queries each, grouped train/dev/test split by
query with leakage checks.
4.2 Corpus and chunking: DPR 100-word Wikipedia passages (fixed chunking); 500k-passage sub-corpus
with full-corpus BM25 hard negatives.
4.3 Grid: {BM25, dense bge-base, hybrid RRF k=60} × k∈{3,10} × reranker {off, MiniLM-L6 cross-encoder
on a fixed pool of 20} × {Qwen2.5-7B, Qwen2.5-1.5B}, greedy decoding = 24 configs.
4.4 Harness: AutoRAG modules, staged and cached (retrieve once, rerank once, generate per generator).
4.5 Metrics: one per task, pre-committed: token-F1 (+EM) for QA, accuracy for FEVER, EM for zsRE; reported per
task, per dataset, and pooled micro / macro-over-datasets / macro-over-tasks; RAGAS on a validated subsample.
4.5b Routers are trained globally (all tasks pooled) and per task, and compared on the same test questions.
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
Datasets, models, methods and tools. Checked on 2026-10-28 against the ACL Anthology, arXiv, or the DOI record
(OpenAlex), unless marked "software":
9. ✔ Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
   Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
   Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, Slav Petrov. Natural Questions: A Benchmark for Question
   Answering Research. TACL 7:452–466, 2019. doi:10.1162/tacl_a_00276.
10. ✔ Kenton Lee, Ming-Wei Chang, Kristina Toutanova. Latent Retrieval for Weakly Supervised Open Domain Question
    Answering. ACL 2019, pp. 6086–6096. doi:10.18653/v1/P19-1612. (Source of the NQ-open split.)
11. ✔ Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, Christopher D. Manning.
    HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018, pp. 2369–2380.
    doi:10.18653/v1/D18-1259.
12. ✔ Mandar Joshi, Eunsol Choi, Daniel Weld, Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge
    Dataset for Reading Comprehension. ACL 2017, pp. 1601–1611. doi:10.18653/v1/P17-1147.
13. ✔ James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal. FEVER: a Large-scale Dataset for Fact
    Extraction and VERification. NAACL 2018, pp. 809–819. doi:10.18653/v1/N18-1074.
14. ✔ Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
    Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, Sebastian Riedel.
    KILT: a Benchmark for Knowledge Intensive Language Tasks. NAACL 2021, pp. 2523–2544.
    doi:10.18653/v1/2021.naacl-main.200. (Source of our FEVER split.)
15. ✔ Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.
    Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020, pp. 6769–6781.
    doi:10.18653/v1/2020.emnlp-main.550. (Source of the 21M-passage Wikipedia corpus.)
16. ✔ Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and
    Individual Rank Learning Methods. SIGIR 2009, pp. 758–759. doi:10.1145/1571941.1572114.
17. ✔ Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, Jian-Yun Nie. C-Pack: Packed Resources For
    General Chinese Embeddings. SIGIR 2024. arXiv:2309.07597. (Paper for the BGE embedding models;
    we use `BAAI/bge-base-en-v1.5`.)
18. ✔ Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019,
    pp. 3982–3992. doi:10.18653/v1/D19-1410. (Library behind our cross-encoder; we use
    `cross-encoder/ms-marco-MiniLM-L-6-v2`.)
19. ✔ Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara,
    Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, Tong Wang. MS MARCO: A Human
    Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268, 2016. (Training data of the cross-encoder.
    Cite as Bajaj et al. for this arXiv version: its author list is alphabetical.)
20. ✔ Qwen Team (An Yang et al.). Qwen2.5 Technical Report. arXiv:2412.15115, 2024.
21. ✔ Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang,
    Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023,
    pp. 611–626. doi:10.1145/3600006.3613165. (Cite only if the vLLM backend is used.)
22. ✔ Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, Rodrigo Nogueira. Pyserini: A Python
    Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations. SIGIR 2021,
    pp. 2356–2362. doi:10.1145/3404835.3463238. (The arXiv preprint 2102.10073 has a different title, "... An
    Easy-to-Use Python Toolkit to Support Replicable IR Research ...". Cite the SIGIR version.)
23. ✔ Jacob Cohen. A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement 20(1):37–46,
    1960. doi:10.1177/001316446002000104.
24. ✔ Philipp Koehn. Statistical Significance Tests for Machine Translation Evaluation. EMNLP 2004, pp. 388–395.
    (Anthology W04-3250; no DOI.)
25. ✔ Taylor Berg-Kirkpatrick, David Burkett, Dan Klein. An Empirical Investigation of Statistical Significance in NLP.
    EMNLP-CoNLL 2012, pp. 995–1005. (Anthology D12-1091; no DOI.)
Added 7 Nov 2026 from the citation check (docs/11). Abstracts read; metadata from arXiv or the DOI record (OpenAlex):
26. ✔ Melissa Z. Pan, Negar Arabzadeh, Mathew Jacob, Fiodar Kazhamiaka, Esha Choukse, Matei Zaharia. Natural Language
    Query to Configuration for Retrieval Agents (BRANE). arXiv:2605.27361, 2026.
27. ✔ Ankit Srivastava. RCT-MARS: When Per-Query Retrieval Routing Fails, and What It Takes to Succeed. Journal of
    Intelligent Information Systems, 2026. doi:10.1007/s10844-026-01080-5.
28. ✔ Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia. Can QPP Choose the Right Query Variant?
    Evaluating Query Variant Selection for RAG Pipelines. SIGIR 2026. doi:10.1145/3805712.3808571.
29. ✔ Steve Cronen-Townsend, Yun Zhou, W. Bruce Croft. Predicting Query Performance. SIGIR 2002, pp. 299–306.
    doi:10.1145/564376.564429.
30. ✔ Yun Zhou, W. Bruce Croft. Query Performance Prediction in Web Search Environments. SIGIR 2007, pp. 543–550.
    doi:10.1145/1277741.1277835.
31. ✔ Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber, Gad Markovits. Predicting Query Performance by Query-Drift
    Estimation. ACM TOIS 30(2):1–35, 2012. doi:10.1145/2180868.2180873.
32. ✔ Ziqi Wang, Xi Zhu, Shuhang Lin, Haochen Xue, Minghao Guo, Yongfeng Zhang. RAGRouter-Bench: A Dataset and Benchmark
    for Adaptive RAG Routing. arXiv:2602.00296, 2026.
33. ✔ Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Guihai Chen. RAGRouter: Learning to Route Queries to
    Multiple Retrieval-Augmented Language Models. NeurIPS 2025. arXiv:2505.23052.
34. ✔ Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie. Fast or Better? Balancing Accuracy and Cost in
    Retrieval-Augmented Generation with Flexible User Control. arXiv:2502.12145, 2025.
35. ✔ Pin Qian, Su Wang, Chong Peng, Junxian You, Lifei Liu, Haoran Yu, Yihang Chen, Xiaochong Jiang. When Should Active
    RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost. KDD 2026 Workshop on Evaluation and
    Trustworthiness of Agentic AI. arXiv:2607.24010.
36. ✔ Matthew Barker, Andrew Bell, Evan Thomas, James Carr, Thomas Andrews, Umang Bhatt. Faster, Cheaper, Better:
    Multi-Objective Hyperparameter Optimization for LLM and RAG Systems. arXiv:2502.18635, 2025.
37. ✔ Y. K. Zhou, Zixuan Wang, Yixiang Fang. HAMMER: An Automatic RAG Tuning System via Hierarchical Memory-Guided
    Monte Carlo Tree Search. Proceedings of the ACM on Management of Data, 2026. doi:10.1145/3802071.
38. ✔ Clovis Varangot-Reille, Christophe Bouvard, Mathieu Ciancone, Antoine Gourru, Marion Schaeffer, François Jacquenet
    (et al.). Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language
    Model-Based Systems. JAIR, 2026. doi:10.1613/jair.1.19801.
39. ✔ Mateusz Śmigielski, Michał Rajkowski, Mateusz Zbrocki, Michał Bernacki-Janson, Karol Kunicki, Julianna
    Godziszewska (et al.). Chunking Methods on Retrieval-Augmented Generation – Effectiveness Evaluation Against
    Computational Cost and Limitations. Procedia Computer Science, 2026. doi:10.1016/j.procs.2026.10.097.
40. ✔ Timur Galimzyanov, Olga Kolomyttseva, Egor Bogomolov. Practical Code RAG at Scale: Task-Aware Retrieval Design
    Choices under Compute Budgets. arXiv:2510.20609, 2025.
41. ✔ Meta Llama Team (Aaron Grattafiori et al.). The Llama 3 Herd of Models. arXiv:2407.21783, 2024. (Judge model:
    Llama-3.1-8B-Instruct, 4-bit AWQ checkpoint `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`.)
Software (cite by URL and the exact version we ran; versions from the Ada job logs `env_gridB_43945*.txt`):
AutoRAG 0.3.27 (paper = ref. 2); llama.cpp, github.com/ggml-org/llama.cpp, release b11936 (commit 27b7a8a);
Qwen2.5-1.5B/7B-Instruct Q8_0 GGUF weights (Hugging Face, Qwen); chromadb 1.5.9; rank-bm25 0.2.2;
sentence-transformers 6.2.0; transformers 5.21.0; torch 2.14.1+cu126; scikit-learn 1.9.1; Pyserini 1.6.0.
