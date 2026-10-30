# Deliverable 3: SLURM on Ada

All scripts are in `slurm/`. Replace `VERIFY_ACCOUNT` / `VERIFY_PARTITION` and the paths in
`slurm/common.sh` once (`grep -rn VERIFY slurm/`). Check with the Ada documentation or admins: GPU type
per partition, `--gres` syntax, max wall time, max GPUs per user, scratch paths, `--exclusive`
policy, and whether compute nodes reach the internet.

**GPU floor (verified):** vLLM needs compute capability ≥ 7.5. Request RTX 2080 Ti or newer
(e.g. `--gres=gpu:<type>:2` if Ada names GPU types; VERIFY the type string). GTX 1080 Ti nodes will fail.

## Ada facts for account `cs4406.11` (checked 2026-10-27)
| Item | Value | Consequence |
|---|---|---|
| Login | `ada-gw2.iiit.ac.in` (password OK; at times refuses with "Not allowed at this time"). `ada.iiit.ac.in` accepts only Kerberos. | Use the gateway. |
| Account / QoS / partition | `cs4406` / `cs4406` / `u22` only | `#SBATCH -A cs4406 -p u22` |
| QoS limits | **1 GPU, 10 CPUs per user; 1 running + 1 submitted job; 6 h wall time** | No job arrays, no `--dependency` chains, no TP=2. Every stage must resume from its cache within 6 h. |
| GPU seen | gnode002: **GTX 1080 Ti, 11 GB, sm_61**. SLURM records no GPU types. | vLLM **and** the installed PyTorch (2.11+cu128, archs sm_75..sm_120) cannot use it. ✔ **RTX 2080 Ti on gnode042 onwards** (TA, 2026-10-28). vLLM jobs use `--exclude` of the u22 nodes below gnode042 so any free 2080 Ti node can take them, and `require_vllm_gpu` (env.sh) stops a job that lands on an older GPU. |
| 7B generator on 1 GPU | fp16 needs ~15 GB | On an 11 GB 2080 Ti: use `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit, ~5.5 GB) or Qwen2.5-3B; on ≥ 24 GB: fp16/bf16 as planned. Decide once the GPU type is known. |
| Storage | home 25 GB (`/home2/cs4406.11`); no personal `/share1` dir; node-local `/ssd_scratch` (~438 GB free) and `/scratch` (~540 GB free) | Envs and code in home; corpus, indexes and HF model cache in node-local scratch, staged inside each job (re-download if the job lands on another node). |
| Internet on compute nodes | yes (Hugging Face, PyPI) | No offline prefetch needed. |
| CUDA | driver 570 (CUDA 12.8); modules `u22/cuda/9.2 … 12.9` | Use cu128 PyTorch wheels. |
| Java | not available as a module | Install OpenJDK 21 via micromamba for Pyserini. |
| Environments | micromamba at `~/micromamba`, env `ragroute` (Python 3.11); code snapshot in `~/ragroute_run/code` | Run with `micromamba run -n ragroute ...` |

The sbatch scripts in `slurm/` were written before these limits were known (they use arrays, 2 GPUs and
dependency chains). They will be rewritten as single-job, 1-GPU, ≤ 6 h scripts once the GPU type is settled.

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

**Cost-weight calibration (job K, before the full run).** After the engine decision (vLLM vs llama.cpp), run
`sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobK_calibrate_gpu.sbatch` on an RTX 2080 Ti node (≤ 3 h,
1 GPU). It times generation at 3 prompt sizes × 2 forced output lengths × 3 repeats per generator, and the
reranker, all on the pilot's cached passages. It then fits the weights into `configs/cost_weights_measured.yaml`.
Commit that file. From then on, `collect` uses the measured weights. Set `RAGROUTE_BACKEND=llamacpp
RAGROUTE_ENV=ragroute` to calibrate llama.cpp instead.

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
