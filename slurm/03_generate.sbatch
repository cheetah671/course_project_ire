#!/bin/bash
#SBATCH --job-name=rr-generate
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
# One array task per generator. Each task loads its model ONCE and runs all 12 retrieval variants.
# qwen7b uses TP=2 (2 x 11 GB cards, fp16). qwen1p5b needs 1 GPU; it is given 2 only to keep one script.
# Every config writes its own cache file, so a job killed by the time limit resumes where it stopped.
source slurm/common.sh
conda activate ragroute
GENS=(qwen1p5b qwen7b)
GEN=${GENS[$SLURM_ARRAY_TASK_ID]}
log_env "generate_${GEN}"
python -m ragroute.harness.run_stages generate --generator "$GEN"
