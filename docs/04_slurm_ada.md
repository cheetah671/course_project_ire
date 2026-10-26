# Deliverable 3: SLURM on Ada

All scripts are in `slurm/`. Replace `VERIFY_ACCOUNT` / `VERIFY_PARTITION` and the paths in
`slurm/common.sh` once (`grep -rn VERIFY slurm/`). Check with the Ada documentation or admins: GPU type
per partition, `--gres` syntax, max wall time, max GPUs per user, scratch paths, `--exclusive`
policy, and whether compute nodes reach the internet.

| Script | Resources (planning) | What it does | Resumable? |
|---|---|---|---|
| `00_setup_env.sh` | login node | 3 conda envs, model/dataset/corpus prefetch, Pyserini index download | n/a |
| `01_build_corpus.sbatch` | 20 CPU, 12 h | BM25 mining (Pyserini), sub-corpus, splits, AutoRAG YAMLs | rerun is idempotent |
| `01b_dense_mining.sbatch` | array 0–7 × 1 GPU, 6 h | optional full-corpus dense mining | per shard |
| `02_retrieval_rerank.sbatch` | 1 GPU, 24 h | AutoRAG retrieval trial (ingest + 3 retrievers), rerank, features, offline cost | stage-level cache |
| `03_generate.sbatch` | array 0–1 × 2 GPUs, 24 h | one task per generator; all 12 retrieval variants | config-level cache |
| `04_checks.sbatch` | 1 GPU, 6 h | equivalence check, token-count check, greedy reproducibility | yes |
| `05_collect_router.sbatch` | 20 CPU, 8 h | score grid, routers, tables, figures | yes |
| `06_latency.sbatch` | 2 GPUs, exclusive, 24 h | latency study, both generators in one session | rerun = new session |
| `07_judge.sbatch` | 2 GPUs, 24 h | vLLM judge server + RAGAS (`MODE=validate` or `full`) | rerun |

`scripts/run_pipeline.sh` submits 01 → 02 → 03 → 05 with `--dependency=afterok`.

## Queue and time-limit strategy
- Every expensive stage writes one file per unit (per retriever, per config) and skips finished units,
  so a job killed at the wall-time limit loses at most one unit. Resubmit the same script.
- Pilot first (week 1): `subcorpus_size: 100000`, `n: 100` per dataset. It measures prefill
  throughput, BM25 seconds/query, and token counts, which replace the planning numbers in
  `scripts/budget.py` and `configs/cost_weights.yaml`.
- The 7B model needs 2 × 11 GB GPUs (TP=2, fp16). If 2-GPU jobs queue for long, alternatives:
  AWQ-quantised 7B on one GPU (changes quality: report it), or run the 1.5B generator first so
  routing code can be developed on half the grid.

## Reproducibility checklist per job
- `results/env/<tag>_<jobid>.txt` (GPU name, driver, git commit, Python) and `_pip.txt` (pip freeze).
- `stage_timings.jsonl` records host, job id, and GPU for every stage.
- Seeds fixed in `configs/experiment.yaml`; `PYTHONHASHSEED=0`.
- Never compare wall-clock numbers across jobs. Only the latency study (one job) reports timing as a result.
