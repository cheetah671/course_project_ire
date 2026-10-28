#!/bin/bash
#SBATCH -J rr-gridB
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH -o /home2/cs4406.11/ragroute_run/logs/%x_%j.out
# Job B (1 GPU): AutoRAG retrieval trial (BM25 + Chroma ingest, bm25/dense/hybrid top-20), cross-encoder rerank,
# router features, generation for both generators (backend from configs/experiment.yaml: llamacpp on GTX 1080 Ti),
# then collect + score the grid. Each stage caches per unit, so resubmitting after the 6 h limit resumes.
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobB_gpu_grid.sbatch
set -euo pipefail
source /home2/cs4406.11/ragroute_run/code/slurm/ada/env.sh
cd "$CODE"
log_env gridB
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

step "retrieval (AutoRAG Evaluator; ingest + 3 retrievers)"
py -m ragroute.harness.run_stages retrieval
step "rerank (cross-encoder, pool of 20)"
py -m ragroute.harness.run_stages rerank
step "router features"
[ -s "$RESULTS_ROOT/features/retrieval_features.parquet" ] || py -m ragroute.features.build_features --device cuda

for GEN in qwen1p5b qwen7b; do
  step "generate $GEN (all 12 retrieval variants)"
  py -m ragroute.harness.run_stages generate --generator "$GEN"
done

step "collect + score the grid"
py -m ragroute.harness.collect
step "DONE"
