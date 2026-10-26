# Deliverable 5: Tables, figures, analysis plan, final report template

Every table and figure is produced by code (`analysis/tables.py`, `analysis/plots.py`,
`router/train_eval.py`), from result files only. Placeholders: `[[...]]`.

## A. Tables
| ID | Title | Rows × columns | Source |
|---|---|---|---|
| T1 | Datasets and corpus | dataset × {source split, size (VERIFY), sampled n, train/dev/test, gold-mapping rate}; corpus: full passages, sub-corpus composition (mined / gold neighbours / random) | `corpus_stats.json`, `leakage_report.json` |
| T2 | Static grid | 24 configs × dataset: EM, F1 [95% CI], input tokens, output tokens, rerank pairs, index queries, scalar cost, cost CV | `T2_static_grid.csv` |
| T3 | Oracle ceiling | per dataset + ALL: best static (chosen on train), oracle quality, headroom [CI], all-fail / all-succeed / mixed shares, #distinct oracle-best configs, top-label share | `T3_oracle_vs_static.csv` |
| T3b | Oracle vs permutation null | real oracle, null mean, null 95th pct, p, predictable-headroom upper bound | `T3b_oracle_null_permutation.csv` |
| T4 | **Headline: routers vs best static** | {Router A, Router B (best model), Adaptive-RAG-style, per-dataset static, random, cheapest, oracle} × {iso-cost ΔQ [CI], p; iso-quality cost saving % [CI on ΔQ]; gap closed} | `headline.csv` |
| T5 | Ablations | feature set {A_noDS, A, A_noEmb, B, B_noDS, D_only, R_only} × model {LR, GBDT, MLP}: gap closed (mean ± seed SD), iso-cost ΔQ | `headline.csv` |
| T6 | Where routing helps / does not | Router B vs static by dataset, hops, HotpotQA type: n, ΔQ [CI], p (Holm-adjusted) | `T6_breakdown_*.csv` |
| T7 | Best config per question type | dataset × qtype: best, runner-up, win frequency, margin [CI], claim supported? | `T7_best_by_qtype.csv` |
| T8 | Cost-weight sensitivity | weight set × {Kendall τ of cost ranking, frontier Jaccard, router iso-cost ΔQ} + summary over 500 draws | `T8_*.csv`, reruns |
| T9 | Offline vs marginal cost | config × amortisation factor {1e3…1e6}: marginal, amortised offline, total; index build time/size | `T9_offline_vs_marginal.csv` |
| T10 | Latency (secondary) | config: p50, p95 (one session; GPU named in caption) | `T10_latency.csv` |
| T11 | Judge validation | IAA κ, judge–human Spearman/Pearson/κ [CIs], EM false-negative rate | `judge_validation_report.json` |
| T12 | RAGAS on subsample | config × {faithfulness, answer relevancy} (only if T11 passes the rule) | `ragas_scores.parquet` |
| A1 | Equivalence & reproducibility (appendix) | retrieved-id agreement, answer agreement, greedy reproducibility, token-count check | `equivalence_*.json`, `greedy_reproducibility.json` |

## B. Figures
| ID | Figure | Encoding |
|---|---|---|
| F1 | Method diagram: grid → oracle table → router (A: question; B: + probe) → Pareto evaluation | drawn by hand (draw.io), not data |
| F2 | **Cost–quality Pareto, test, all datasets**: 24 static points (grey), static frontier, per-dataset static, Adaptive-RAG-style (grey dashes), Router A (blue), Router B (orange), oracle (aqua); direct labels; step curves | `F2_pareto_test.pdf` |
| F3 | Same, one panel per dataset (small multiples, shared y) | `F3_pareto_by_dataset.pdf` |
| F4 | Oracle-best share per config × hops type (heatmap, sequential blue) | `F4_oracle_winrate.pdf` |
| F5 | Gap closed by feature set × model (bars; seed SD as error bars, to add) | `F6_ablation_gap_closed.pdf` |
| F6 | GBDT gain by feature group (100% stacked bar) | `F7_importance.pdf` |
| F7 | Weight sensitivity: distribution of Kendall τ and of router ΔQ over 500 draws (histograms) | to add from T8 |
| F8 | Amortisation: total cost per query vs queries per build, dense vs BM25 configs (log x) | to add from T9 |

Palette: three validated slots (#2a78d6, #eb6834, #1baf7a; passes all-pairs CVD checks) for the three
highlighted series; everything else neutral grey with dash style plus a legend entry. Aqua has <3:1
contrast, so every coloured curve carries a direct text label.

## C. Analysis plan (in the order the report uses it)
1. **Sanity:** equivalence (A1), gold-mapping rates (T1), leakage report, greedy reproducibility.
2. **Static landscape (T2, F2 points):** which configs sit on the static frontier; paired bootstrap of
   each config vs the best static (Holm within the family).
3. **Oracle (T3, T3b, F4):** headroom vs best static; how much survives the permutation null; label
   statistics (if > [[x]]% of queries are all-fail or all-succeed, routing has little room by construction).
4. **Routers (T4, F2, F3):** iso-cost and iso-quality comparisons on test with λ chosen on dev; gap closed.
5. **Ablations (T5, F5, F6):** does Router B beat A after charging probe cost (reuse and no-reuse)?
   How much is dataset identity (`D_only`, per-dataset static)?
6. **Where it helps / does not (T6, T7):** per dataset and question type, with CIs; only CI-backed claims.
7. **Robustness (T8, T9, F7, F8):** cost weights, amortisation, chance-corrected FEVER, micro vs macro.
8. **Secondary (T10–T12):** latency, judge validation, RAGAS.
9. **Both outcomes are planned for:**
   - *Router wins* (gap closed CI > 0): emphasise the cost saving at iso-quality and which signals matter.
   - *Small gain* (CI includes 0, or headroom ≈ null): lead with T3b. A narrow predictable ceiling
     explains why static best-practice recipes work; report the regions where routing helps, if any,
     and the upper bound on what any router could gain on these benchmarks.

## D. Final report template (8 pages + appendix; adapt to the course format)
```
Title / authors / abstract (version W or S, docs/01_proposal.md)
1 Introduction                        (motivation, RQ1–RQ5, contributions)          ~0.75 p
2 Related work                        (positioning table, compressed)               ~0.5 p
3 Setup                                                                             ~1.5 p
  3.1 Data and corpus (T1; fixed chunking; sub-corpus)  3.2 Grid and harness (F1)
  3.3 Metrics and normalisation        3.4 Cost model (offline vs marginal)
4 Oracle labelling: the static landscape and the ceiling (T2 excerpt, T3, T3b, F4)  ~1 p
5 Routing (method: formulation, features, baselines, λ selection)                  ~0.75 p
6 Results                                                                           ~2 p
  6.1 Headline (T4, F2)  6.2 Router A vs B and the price of signals (T5, F5, F6)
  6.3 Where routing helps and where it does not (T6, T7, F3)
  6.4 Robustness: cost weights and amortisation (T8, T9, F7, F8)
7 Secondary: latency and LLM-judge metrics (T10–T12)                                ~0.5 p
8 Discussion: METIS-style online adaptation; what a narrow ceiling means; limitations
  (fixed chunking, sub-corpus, two generators, English Wikipedia QA only, EM strictness) ~0.75 p
9 Conclusion                                                                        ~0.25 p
Appendix: full T2, equivalence/reproducibility (A1), annotation guide, leakage report, hyper-parameters,
          environment (GPU type, library versions), all VERIFY items resolved.
```
