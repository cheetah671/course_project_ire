# Deliverable 2: Repository and runnable code

## Layout
```
configs/
  experiment.yaml            single source of truth: seeds, datasets, grid, models, prompts, judge
  cost_weights.yaml          cost weights (default + sensitivity sets), amortisation factors
  autorag/                   GENERATED on the cluster by make_autorag_yaml (gitignored)
src/ragroute/
  config.py                  config loading (${VAR:-default} expansion), RagConfig, 24-config grid
  data/prepare_qa.py         load + normalise + seeded (stratified) sampling     -> qa_raw.parquet
  data/mine_candidates.py    full-corpus BM25 top-100 via Pyserini              -> mined_bm25.parquet
  data/mine_dense.py         optional full-corpus dense mining (sharded)        -> mined_dense.parquet
  data/build_corpus.py       sub-corpus + AutoRAG corpus.parquet / qa.parquet + qa_meta.parquet
  data/splits.py             grouped 60/20/20 split + leakage report            -> splits.parquet
  harness/make_autorag_yaml.py  retrieval-stage YAML + 24 one-module-per-node config YAMLs
  harness/run_stages.py      staged cached runner: retrieval (AutoRAG Evaluator) -> rerank -> generate
  harness/check_equivalence.py  staged runner vs AutoRAG end-to-end on 50 queries
  harness/collect.py         merge + score + cost                               -> runs_scored.parquet (oracle table)
  eval/metrics.py            EM, token-F1, verdict accuracy, answer extraction
  eval/ragas_eval.py         RAGAS faithfulness / answer relevancy on a subsample
  eval/judge_validation.py   150-item sample -> annotation sheet; agreement statistics
  eval/latency.py            same-session latency study
  features/query_features.py, retrieval_features.py, build_features.py
  oracle.py                  oracle policies, label statistics, permutation null
  cost.py                    cost model, router overhead accounting, weight draws
  router/models.py           fractional LR, LightGBM, MLP
  router/policies.py         static/per-group/random baselines, router selection, Pareto helpers
  router/train_eval.py       trains all routers + baselines, λ selection, headline table
  stats/bootstrap.py         paired bootstrap, Holm, best-config stability
  analysis/tables.py, plots.py   report tables (CSV + LaTeX) and figures (PDF + PNG)
  synthetic.py               FAKE data for dry runs (never reported)
slurm/                       Ada scripts (docs/04_slurm_ada.md)
scripts/budget.py            budget arithmetic; record_offline_cost.py; run_pipeline.sh
annotation/                  judge-validation template
tests/                       unit tests + synthetic end-to-end test (19 tests)
```

## Data formats (AutoRAG 0.3.27 schema, checked in its source)
- `corpus.parquet`: `doc_id` (str), `contents` (str), `metadata` (dict; AutoRAG requires
  `last_modified_datetime`; we write a fixed 2018-12-20 timestamp for reproducibility; AutoRAG adds
  `prev_id`/`next_id`).
- `qa.parquet`: `qid` (str), `query` (str), `retrieval_gt` (list of lists of doc_ids: outer = AND
  groups, inner = OR alternatives; `[[]]` allowed), `generation_gt` (list of str). Extra metadata lives in
  `qa_meta.parquet` so AutoRAG sees exactly its schema.

## Commands (in order)
```bash
python -m ragroute.data.prepare_qa
python -m ragroute.data.mine_candidates --threads 16        # env ragroute-mining
python -m ragroute.data.build_corpus
python -m ragroute.data.splits
python -m ragroute.harness.make_autorag_yaml
python -m ragroute.harness.run_stages retrieval            # AutoRAG Evaluator.start_trial(retrieval_stage.yaml)
python -m ragroute.harness.run_stages rerank
python -m ragroute.features.build_features --device cuda
python -m ragroute.harness.run_stages generate --generator qwen1p5b
python -m ragroute.harness.run_stages generate --generator qwen7b
python -m ragroute.harness.collect
python -m ragroute.router.train_eval
python -m ragroute.analysis.tables && python -m ragroute.analysis.plots
```
Equivalent AutoRAG CLI for one config (used only for the equivalence check; Python API shown in code):
`autorag evaluate --config configs/autorag/full/<config_id>.yaml --qa_data_path ... --corpus_data_path ... --project_dir ...`

## Example generated config (hybrid, k=3, reranker on, 7B)
```yaml
node_lines:
- node_line_name: retrieve_node_line
  nodes:
  - {node_type: lexical_retrieval, top_k: 20, strategy: {metrics: [retrieval_recall, retrieval_ndcg, retrieval_mrr]},
     modules: [{module_type: bm25, bm25_tokenizer: porter_stemmer}]}
  - {node_type: semantic_retrieval, top_k: 20, strategy: {metrics: [...]},
     modules: [{module_type: vectordb, vectordb: bge_base_chroma}]}
  - {node_type: hybrid_retrieval, top_k: 20, strategy: {metrics: [...]},
     modules: [{module_type: hybrid_rrf, weight_range: (60, 60)}]}
- node_line_name: post_retrieve_node_line
  nodes:
  - {node_type: passage_reranker, top_k: 3, strategy: {metrics: [...]},
     modules: [{module_type: sentence_transformer_reranker, model_name: cross-encoder/ms-marco-MiniLM-L-6-v2}]}
  - {node_type: prompt_maker, strategy: {metrics: [bleu, rouge]}, modules: [{module_type: chat_fstring, prompt: [[...]]}]}
  - {node_type: generator, strategy: {metrics: [bleu, rouge]},
     modules: [{module_type: vllm, llm: Qwen/Qwen2.5-7B-Instruct, temperature: 0.0, max_tokens: 32,
                tensor_parallel_size: 2, dtype: half, max_model_len: 4096}]}
vectordb:
- {name: bge_base_chroma, db_type: chroma, client_type: persistent, collection_name: bge_base,
   embedding_model: [{type: huggingface, model_name: BAAI/bge-base-en-v1.5,
                      query_instruction: "Represent this sentence for searching relevant passages: "}],
   similarity_metric: cosine, path: <project>/resources/chroma}
```

## Caching and reproducibility
- Stage outputs are files; each stage skips existing outputs (`--force` to redo). Index ingestion lives
  in the AutoRAG project dir and is reused by every later stage and by the equivalence check.
- Seeds: global 20261026 (sampling, split, random fill); router seeds 0–4; vLLM sampling seed 0
  (greedy anyway). `PYTHONHASHSEED=0`.
- Each SLURM job writes `results/env/<tag>_<job>.txt` (GPU name, driver, git commit, Python) and a
  `pip freeze`; every timing record carries host, job id and GPU name.

## Tested locally
`pytest`: 19 tests (metrics, bootstrap, split leakage guard, oracle ties, cost accounting, and a
synthetic end-to-end run of routers, tables and figures). The AutoRAG/vLLM stages cannot run without
the cluster; they are covered by the week-1 pilot and `check_equivalence`.
