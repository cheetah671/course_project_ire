#!/bin/bash
#SBATCH --job-name=rr-latency
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --exclusive                 # VERIFY allowed on Ada; reduces interference from co-located jobs
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# SECONDARY latency study. ONE job = ONE session: both generators run here, sequentially, on the same node.
# Never merge latency numbers from different jobs.
source slurm/common.sh
conda activate ragroute
log_env latency
SESSION="lat_${SLURM_JOB_ID}"
python -m ragroute.eval.latency --generator qwen1p5b --session "$SESSION" --n 200 --repeats 3
python -m ragroute.eval.latency --generator qwen7b   --session "$SESSION" --n 200 --repeats 3
