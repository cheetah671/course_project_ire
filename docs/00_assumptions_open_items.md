# Assumptions and open items

Read this first. Each assumption says what happens if it turns out false. "VERIFY" means it must be
checked against the real source (docs, model card, cluster admin) before we rely on it.

## A. Assumptions

| # | Assumption | If false |
|---|---|---|
| A1 | Team of 3, ~8 weeks, starting Mon 2026-10-26; final report due around 2026-12-20. **VERIFY deadline.** | Compress weeks 6–8 (drop the latency study and FEVER first). |
| A2 | Ada GPUs are 11 GB RTX 2080 Ti-class (Turing, no bf16), 4 GPUs per node, with per-job time limits around a day or more. **VERIFY GPU type, `--gres` syntax, partition/account names, max wall time, scratch paths.** | On ≥24 GB Ampere cards: TP=1 for the 7B model, `dtype: bfloat16`, budget roughly halves. |
| A3 | Compute nodes have no internet, so models and datasets are prefetched on a login node into `$HF_HOME`. | If they do have internet, nothing changes; the offline flags just become unnecessary. |
| A4 | AutoRAG **0.3.27** (latest on PyPI at writing) is the harness. Its behaviour below was checked by reading its source, not by running it. **VERIFY by the week-1 equivalence check.** | If a later version changes node semantics, pin 0.3.27. |
| A5 | Open-weight generators from one family: **Qwen2.5-7B-Instruct** and **Qwen2.5-1.5B-Instruct** (same tokenizer, so token counts are comparable). **VERIFY licences and parameter counts on the model cards.** | Llama-3.1-8B / 3.2-3B are drop-in alternatives (gated licences, different tokenizers). |
| A6 | Corpus: DPR Wikipedia split (2018-12-20 dump, 100-word passages, `id/text/title` TSV, ~21.0M passages). **VERIFY URL, count, format.** | KILT knowledge source (2019-08-01) is the alternative (see design doc D1). |
| A7 | ~1,000 queries per dataset, taken from the public dev/validation splits (we train nothing on them except the router, which uses its own grouped split). | — |
| A8 | No paid API is used for generation. API prices appear only as an illustrative reference and are marked VERIFY. | If the course provides API credit, it could fund a stronger judge. |
| A9 | "Cost" means countable marginal cost (tokens, calls, reranker pairs, index queries) with documented weights. Latency is secondary and measured in one session only. | — |
| A10 | Group 4 in the spec ("metric: normalised quality and cost for that (query, config)") is the router's **training target**, not an input. Using it as an input feature would leak the label. | — |

## B. "Bllm": unconfirmed term from a team meeting

Not guessed silently. The plan proceeds with the **base design (24 configs, no Bllm)**. Plausible readings:

| Interpretation | What it would add | Change to the plan | Compute impact (per 1,000-query dataset) |
|---|---|---|---|
| **B1. Baseline LLM = closed-book (no retrieval)** | A "no-retrieval" option per generator, as in Adaptive-RAG's no-retrieval branch | +2 configs (26). The router can skip retrieval entirely for easy queries. Cost model: 0 index queries, 0 rerank, ~100 input tokens. | Small: about +0.1 generation GPU-h |
| **B2. Bigger LLM as a 3rd generator** (e.g. Qwen2.5-14B) | Stronger quality ceiling and a wider cost spread | +12 configs (36). 14B in fp16 is ~30 GB, so it needs TP=4 on 11 GB cards, or AWQ quantisation (which changes quality). | About +5–6 GPU-h, roughly 1.8× the 7B share (VERIFY) |
| **B3. Black-box / API LLM** (as generator or judge) | Real dollar cost; a stronger judge | Generator: +12 configs priced in $, not GPU-ms, and not reproducible over time. Judge: replaces the local judge after the same 150-item validation. | $ only; depends on price sheet |
| **B4. LLM-based router baseline** ("LLM decides the config", similar in spirit to METIS's LLM profiler) | One more baseline | Prompt the 7B model with the question and config descriptions; its own LLM call is added to the router's cost. | ~1,000 extra short calls: negligible |
| **B5. Typo, e.g. "BM25 + LLM"** | Nothing new | Already in the grid (bm25-* configs). | — |

**Recommendation if the team confirms any of these:** B1 is cheapest and most informative, because a
no-retrieval option is where per-query routing typically saves the most cost. B2 is next if GPUs allow.
Write the confirmed meaning into this table and into `configs/experiment.yaml`.

## C. Open items (also tracked in docs/09_requirements_checklist.md)

1. Confirm what "Bllm" means (section B).
2. Ada details: GPU type, account/partition, max wall time, scratch paths, whether `--exclusive` is allowed (A2).
3. Final deadline and report format (page limit, template) (A1).
4. Judge model: must not be Qwen; pick after the 150-item validation (candidates: a ~7–9B instruct model from another family; a stronger model if API credit exists).
5. Whether FEVER is included: decided after the week-2 budget check (design doc §3).
6. Pull the reference lists of Wang et al. 2024, AutoRAG, AutoRAG-HP and arXiv:2505.03452, and check who cites them (Semantic Scholar / Google Scholar "cited by"). Add chunking-vs-cost studies found there. **No citation is added without reading the paper.**
7. Confirm the vLLM version that runs fp16 on Ada's GPUs (Turing support in recent vLLM releases: VERIFY).
8. Confirm the Pyserini prebuilt index name and that its docids equal DPR passage ids.
9. Pre-registered judge acceptance threshold (κ ≥ 0.6 proposed): team sign-off.
