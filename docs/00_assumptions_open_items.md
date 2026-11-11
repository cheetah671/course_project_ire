# Assumptions and open items

Read this first. Each assumption says what happens if it turns out false. "VERIFY" means it must be
checked against the real source (docs, model card, cluster admin) before we rely on it.

## A. Assumptions

| # | Assumption | If false |
|---|---|---|
| A1 | ✔ Team of 3. **Mid-submission Fri 2026-10-31; final deadline Fri 2026-11-21** (confirmed by the team). Mid-submission format not yet known. | Plan in docs/07 already cuts scope (latency optional). The 3 tasks (QA, FEVER, zsRE) are never cut. |
| A2 | **Checked on Ada 2026-10-27 (account `cs4406`, QoS `cs4406`, partition `u22`):** max **1 GPU, 10 CPUs, 1 running job and 1 submitted job per user, 6 h wall time**. Node gnode002 has **GTX 1080 Ti (11 GB, compute capability 6.1)**, which vLLM does not support (floor 7.5). GPU types of other nodes are not recorded in SLURM. **TA, 2026-10-28: gnode042 onwards (the `u22` nodes after gnode041, up to about gnode090) have RTX 2080 Ti**; select them with `-w`/`--nodelist` or, as our jobs do, `--exclude` of the u22 nodes below gnode042. | If no vLLM-capable GPU is reachable: see the GPU decision in docs/04 (ask for a GPU allocation, or change the serving engine).
| A3 | ✔ Compute nodes **do** have internet (Hugging Face and PyPI reachable from gnode002). Storage: home quota 25 GB (`/home2/cs4406.11`), no personal `/share1` folder; node-local `/ssd_scratch` (~438 GB free) and `/scratch` (~540 GB free) exist but are per node. | Stage large data (corpus, indexes, models) into node-local scratch inside each job, or ask admins for a `/share1` folder. |
| A4 | AutoRAG **0.3.27** (latest on PyPI at writing) is the harness. Its behaviour below was checked by reading its source, not by running it. **VERIFY by the week-1 equivalence check.** | If a later version changes node semantics, pin 0.3.27. |
| A5 | Open-weight generators from one family: **Qwen2.5-7B-Instruct** and **Qwen2.5-1.5B-Instruct** (same tokenizer, so token counts are comparable). **VERIFY licences and parameter counts on the model cards.** | Llama-3.1-8B / 3.2-3B are drop-in alternatives (gated licences, different tokenizers). |
| A6 | Corpus: DPR Wikipedia split (2018-12-20 dump, 100-word passages, `id/text/title` TSV). ✔ URL live (4.69 GB gz), 21,015,324 passages (matches Pyserini). Still check the TSV header and checksum after download. | KILT knowledge source (2019-08-01) is the alternative (see design doc D1). |
| A7 | ~1,000 queries per dataset, taken from the public dev/validation splits (we train nothing on them except the router, which uses its own grouped split). | — |
| A8 | No paid API is used for generation. API prices appear only as an illustrative reference and are marked VERIFY. | If the course provides API credit, it could fund a stronger judge. |
| A9 | "Cost" means countable marginal cost (tokens, calls, reranker pairs, index queries) with documented weights. Latency is secondary and measured in one session only. | — |
| A10 | Group 4 in the spec ("metric: normalised quality and cost for that (query, config)") is the router's **training target**, not an input. Using it as an input feature would leak the label. | — |

## B. "Bllm" (resolved 2026-10-27)

The team confirmed **"Bllm" meant vLLM**, the LLM inference engine. The base design already serves both
generators (and the judge) with vLLM, so the plan is unchanged: 24 configs, no extra arm.

Consequence: vLLM is a hard requirement, so the generators must run on GPUs vLLM supports
(compute capability ≥ 7.5; see A2 and docs/04).

## C. Open items (updated 11 Nov 2026, after the full run; also tracked in docs/09_requirements_checklist.md)

1. ✔ "Bllm" = vLLM (section B).
2. ✔ Ada: account/partition/QoS limits, scratch, internet (A2, A3). The full run used vLLM 0.11.2 on RTX 2080 Ti
   (gnode042 onwards) on account cs4406.07.
3. ✔ Deadlines known (A1). Open: final report format (pages, template, demo?).
4. ✔ Judge: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (not Qwen). Validated on 150 dev items whose labels
   were LLM-drafted and human-reviewed with the TA's permission; it fails the κ ≥ 0.6 rule, so RAGAS is diagnostic only
   (docs/12 §7).
5. ✔ Tasks: 3 tasks (QA: NQ, HotpotQA, TriviaQA; FEVER; zsRE), routers global and per task. **Still open:** confirm with
   the professor that 3 tasks and the normalisation (each metric on its own 0–1 scale; pooled results also macro over
   tasks, shown in docs/12 §6.1) are what was meant.
6. ✔ Reference lists and citing papers checked: 909 citing papers screened (docs/11); BRANE and RCT-MARS added.
7. ✔ vLLM 0.11.2 runs on the 2080 Ti; fp16 kept for the 1.5B model after the fp32 check (0.862 vs 0.860).
8. ✔ Superseded: we built our own Lucene BM25 index from the DPR TSV, so hit.docid equals the DPR id by construction.
9. ✔ Judge acceptance threshold κ ≥ 0.6 and Spearman CI lower bound > 0.4 applied as pre-registered.
10. ✔ Cost weights measured on the full run's engine and GPU (jobs K, K2; `configs/cost_weights_measured.yaml`).
