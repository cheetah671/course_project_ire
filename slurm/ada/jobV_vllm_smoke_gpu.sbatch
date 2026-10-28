#!/bin/bash
#SBATCH -J rr-vllm-smoke
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=02:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
# RTX 2080 Ti nodes are gnode042 onwards (TA, 2026-10-28), so any free one will do. Slurm rejects unknown names in
# --exclude, hence the explicit list (u22 nodes as of 2026-10-28: sinfo -p u22 -o %N).
#SBATCH -o /home2/cs4406.11/ragroute_run/logs/%x_%j.out
# Job V (1 GPU, RTX 2080 Ti): does vLLM work for both generators on Turing, and does it agree with the pilot?
# Uses the pilot's cached retrieval/rerank, writes only to new cache folders (generation_vllm*) and a separate
# results folder, so the pilot's outputs are untouched.
#  1) input-token count == vLLM's prompt length, per generator
#  2) generate 2 configs x 300 pilot questions per generator; compare with the pilot's llama.cpp outputs
#  3) greedy reproducibility: rerun the same configs, compare
# Usage:  sbatch --export=ALL,RAGROUTE_PILOT=1[,GENERATORS=qwen7b] slurm/ada/jobV_vllm_smoke_gpu.sbatch
set -euo pipefail
export RAGROUTE_ENV=ragroute-vllm RAGROUTE_BACKEND=vllm
source /home2/cs4406.11/ragroute_run/code/slurm/ada/env.sh
export RESULTS_ROOT=$RUN/pilot/results_vllm_smoke
mkdir -p "$RESULTS_ROOT"
cd "$CODE"
require_vllm_gpu
log_env vllmsmoke

for G in ${GENERATORS:-qwen1p5b qwen7b}; do   # GENERATORS=qwen7b reruns one model only
  ONLY=bm25-k10-rr0-$G,dense-k3-rr0-$G
  step "$G: token count check"
  py -m ragroute.harness.run_stages check_tokens --generator $G
  step "$G: generate $ONLY"
  py -m ragroute.harness.run_stages generate --generator $G --only $ONLY --subdir generation_vllm --force
  step "$G: greedy rerun"
  py -m ragroute.harness.run_stages generate --generator $G --only $ONLY --subdir generation_vllm_repro --force
done

step "vLLM vs pilot llama.cpp"
py -m ragroute.harness.compare_generations --a generation --b generation_vllm --out "$RESULTS_ROOT/vllm_vs_llamacpp.json"
step "vLLM greedy reproducibility"
py -m ragroute.harness.compare_generations --a generation_vllm --b generation_vllm_repro \
   --out "$RESULTS_ROOT/vllm_reproducibility.json"
step "DONE"; cat "$RESULTS_ROOT"/token_count_check_*.json
