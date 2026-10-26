#!/bin/bash
# Sourced by every sbatch script. Edit the VERIFY lines once for your Ada account.
set -euo pipefail

export REPO_DIR="${REPO_DIR:-$HOME/course_project_ire}"
# Ada: node-local SSD scratch is fast but wiped after the job; shared storage persists. VERIFY both paths.
export SHARE_DIR="${SHARE_DIR:-/share1/$USER/ragroute}"        # VERIFY: persistent shared storage path
export LOCAL_SCRATCH="${LOCAL_SCRATCH:-/ssd_scratch/$USER}"    # VERIFY: node-local scratch path
export DATA_ROOT="${DATA_ROOT:-$SHARE_DIR/data}"
export RESULTS_ROOT="${RESULTS_ROOT:-$SHARE_DIR/results}"

# Compute nodes may have no internet: download models on a login node first (scripts/prefetch_models.sh)
export HF_HOME="${HF_HOME:-$SHARE_DIR/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# module load cuda/12.1   # VERIFY: `module avail cuda`; vLLM wheels bundle their CUDA runtime but need a recent driver
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"   # VERIFY

cd "$REPO_DIR"
mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$RESULTS_ROOT/env" logs

log_env () {  # reproducibility record per job: GPU type, driver, packages, git commit
  local tag="$1"
  {
    echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "no GPU"
    git rev-parse HEAD 2>/dev/null || echo "no git"
    python --version
  } > "$RESULTS_ROOT/env/${tag}_${SLURM_JOB_ID}.txt"
  pip freeze > "$RESULTS_ROOT/env/${tag}_${SLURM_JOB_ID}_pip.txt"
}
