#!/bin/bash
#SBATCH -J rr-latencyL
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041,060]  # u22 nodes below gnode042; gnode060: CUDA not usable (job 70547)
#SBATCH --constraint=2080ti         # job 70543 landed on gnode042, which has a GTX 1080 Ti (no "2080ti" feature in sinfo)
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job L (1 GPU, after B2): SECONDARY latency study. ONE job = ONE session: BM25 timings, then both generators,
# on the same node. Never merge latency numbers from different jobs. See ragroute.eval.latency.
#   200 test questions (40 per dataset) x 12 configurations x 3 repeats per generator, batch size 1.
# Every finished (generator, repeat, configuration) is saved at once, so a time-out keeps what was measured; the
# summary then covers only the configurations that finished (it says how many).
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
require_vllm_gpu
log_env latencyL
require_py
# Job 70547 (gnode060): nvidia-smi saw the GPU but torch could not initialise CUDA ("CUDA unknown error"); stop at once
py -c "import sys, torch; ok = torch.cuda.is_available(); print('torch cuda_ok', ok); sys.exit(0 if ok else 4)" \
  || { echo "CUDA not usable from torch on $(hostname): resubmit with --exclude adding this node"; exit 4; }
SESSION="lat_${SLURM_JOB_ID}"
echo "session $SESSION"

step "BM25 timings (Lucene over the 500k sub-corpus, single thread)"
pymine -m ragroute.eval.latency bm25 --session "$SESSION" --workdir "$NODE_SCRATCH/index_bench"
for g in qwen1p5b qwen7b; do
  step "pipeline latency: $g"
  py -m ragroute.eval.latency pipeline --session "$SESSION" --generator "$g"
done
step "summary"
py -m ragroute.eval.latency summarise --session "$SESSION"
step "DONE"
