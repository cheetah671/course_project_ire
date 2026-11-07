#!/bin/bash
#SBATCH -J rr-retrB1
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Full run, job B1 (1 GPU, RTX 2080 Ti): everything before generation, then the fp16 check.
#  1) AutoRAG retrieval trial (BM25 + Chroma ingest, bm25/dense/hybrid top-20)
#  2) cross-encoder rerank of each pool
#  3) router features (question embeddings + Router B probe signals)
#  4) precision check for vLLM on Turing (no bf16): the 1.5B model on the first 1,000 questions of one config,
#     once in fp16 and once in fp32, compared with compare_generations. Decides models.generators.qwen1p5b.dtype
#     before job B2 generates the grid. The 7B model is AWQ (4-bit weights, fp16 activations only), so it is not tested.
# Every stage skips finished work, so resubmitting after the 6 h limit resumes.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
require_vllm_gpu
log_env retrB1

step "retrieval (AutoRAG Evaluator; ingest + 3 retrievers)"
py -m ragroute.harness.run_stages retrieval
step "rerank (cross-encoder, pool of 20)"
py -m ragroute.harness.run_stages rerank
step "router features"
[ -s "$RESULTS_ROOT/features/retrieval_features.parquet" ] || py -m ragroute.features.build_features --device cuda

CHECK=dense-k3-rr0-qwen1p5b
for DT in float16 float32; do
  if [ ! -s "$DATA_ROOT/cache/generation_dtype_$DT/$CHECK.parquet" ]; then
    step "precision check: $CHECK in $DT (first 1,000 questions)"
    RAGROUTE_DTYPE=$DT py -m ragroute.harness.run_stages generate --generator qwen1p5b --only $CHECK \
       --limit 1000 --subdir generation_dtype_$DT
  fi
done
step "fp16 vs fp32"
py -m ragroute.harness.compare_generations --a generation_dtype_float32 --b generation_dtype_float16 \
   --out "$RESULTS_ROOT/dtype_check_qwen1p5b.json"
step "DONE"; du -sh "$DATA_ROOT" "$DATA_ROOT"/* 2>/dev/null; df -h "$HOME" | tail -1
