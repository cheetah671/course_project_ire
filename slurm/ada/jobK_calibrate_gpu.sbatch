#!/bin/bash
#SBATCH -J rr-calibrate
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
# RTX 2080 Ti nodes are gnode042 onwards (TA, 2026-10-28): the same GPU type as the full run.
#SBATCH -o /home2/cs4406.11/ragroute_run/logs/%x_%j.out
# Job K (1 GPU): measure the timings behind the cost weights BEFORE the full run, on the full run's GPU and engine.
# Uses the pilot's cached passages (read-only) and writes only to $RUN/calibration and configs/cost_weights_measured.yaml.
#  1) per generator: N prompts x k in {1,3,10} x forced output length {1,32} x 3 repeats (prefix caching off)
#  2) cross-encoder reranker: N queries x 20 pairs x 3 repeats
#  3) fit a_in/a_out/b (per generator), c, d -> configs/cost_weights_measured.yaml (commit it with the results)
# Run AFTER the engine decision (vLLM vs llama.cpp): the weights belong to one engine and GPU type.
# Usage:  sbatch --export=ALL,RAGROUTE_PILOT=1[,GENERATORS=qwen7b] slurm/ada/jobK_calibrate_gpu.sbatch
set -euo pipefail
export RAGROUTE_ENV=${RAGROUTE_ENV:-ragroute-vllm} RAGROUTE_BACKEND=${RAGROUTE_BACKEND:-vllm}
source /home2/cs4406.11/ragroute_run/code/slurm/ada/env.sh
STAGE_TIMINGS=$DATA_ROOT/cache/stage_timings.jsonl     # pilot retrieval/rerank timings (AutoRAG per-query time)
export RESULTS_ROOT=$RUN/calibration
mkdir -p "$RESULTS_ROOT"
cd "$CODE"
[ "$RAGROUTE_BACKEND" = "vllm" ] && require_vllm_gpu
log_env calibrate

for G in ${GENERATORS:-qwen1p5b qwen7b}; do
  step "$G: generation timings ($RAGROUTE_BACKEND)"
  py -m ragroute.harness.calibrate generate --generator $G --n ${N:-200} --repeats 3
done
step "reranker timings"
py -m ragroute.harness.calibrate rerank --n ${N:-200} --repeats 3

step "fit weights"
py scripts/calibrate_costs.py --timings "$RESULTS_ROOT/calibration/calibration_timings.jsonl" \
   --stage-timings "$STAGE_TIMINGS" --out "$CODE/configs/cost_weights_measured.yaml"
step "DONE"; cat "$CODE/configs/cost_weights_measured.yaml"
