#!/bin/bash
#SBATCH -J rr-indexK2
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job K2 (1 GPU, after B2): measure the index-query weight d with standard engines (Lucene BM25; bge-base + Chroma),
# refit configs/cost_weights_measured.yaml, and re-score the grid with it. AutoRAG's own retrieval timing (1,457 ms per
# BM25 query: pure-Python rank_bm25 over 500k passages) measured the harness, not retrieval; it is kept as the
# sensitivity set "measured_autorag_index". See ragroute.harness.calibrate_index.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
require_vllm_gpu
log_env indexK2
require_py
TIMINGS=$RUN/calibration/calibration/calibration_timings.jsonl

if ! grep -q '"component": "index"' "$TIMINGS" 2>/dev/null; then
  step "BM25 index-query timing (Lucene over the 500k sub-corpus)"
  pymine -m ragroute.harness.calibrate_index bm25 --timings "$TIMINGS" --workdir "$NODE_SCRATCH/index_bench"
  step "dense index-query timing (bge-base on GPU + Chroma HNSW)"
  py -m ragroute.harness.calibrate_index dense --timings "$TIMINGS"
fi

step "refit cost weights"
py scripts/calibrate_costs.py --timings "$TIMINGS" --stage-timings "$DATA_ROOT/cache/stage_timings.jsonl" \
   --out "$CODE/configs/cost_weights_measured.yaml"
step "re-score the grid with the refitted weights"
py -m ragroute.harness.collect
grep '"component": "index"' "$TIMINGS" > "$RESULTS_ROOT/index_calibration.json"
step "DONE"; cat "$CODE/configs/cost_weights_measured.yaml" | head -12
