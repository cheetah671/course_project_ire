#!/bin/bash
#SBATCH -J rr-genB2
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Full run, job B2 (1 GPU, RTX 2080 Ti, after B1): generation for all 24 configs with vLLM, then collect + score.
# One parquet per config, so a job killed at 6 h loses at most the config in progress: resubmit until the log
# shows "collect + score the grid". About 1.5 h for the 1.5B model and ~6 h for the 7B AWQ model (5,000 questions).
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
require_vllm_gpu
log_env genB2

for GEN in ${GENERATORS:-qwen1p5b qwen7b}; do
  step "generate $GEN (all 12 retrieval variants)"
  py -m ragroute.harness.run_stages generate --generator "$GEN"
done
step "token-count check (our tokenizer vs vLLM), per generator"
for GEN in qwen1p5b qwen7b; do
  [ -s "$RESULTS_ROOT/token_count_check_$GEN.json" ] || py -m ragroute.harness.run_stages check_tokens --generator "$GEN"
done

step "offline (once-per-corpus) cost: index sizes + timed build slices"
[ -s "$RESULTS_ROOT/offline_cost.json" ] || py scripts/record_offline_cost.py

step "collect + score the grid"
py -m ragroute.harness.collect
step "DONE"
