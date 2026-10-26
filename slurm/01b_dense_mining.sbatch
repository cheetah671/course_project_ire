#!/bin/bash
#SBATCH --job-name=rr-densemine
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
# OPTIONAL (corpus.dense_mining=true): embed the full 21M corpus in 8 shards, then run with --search.
# Submit:  sbatch slurm/01b_dense_mining.sbatch
# Then:    sbatch --dependency=afterok:<jobid> --array=0 --export=ALL,MODE=search slurm/01b_dense_mining.sbatch
# Must finish BEFORE 01_build_corpus.sbatch's build step (build_corpus picks up mined_dense.parquet if present).
source slurm/common.sh
conda activate ragroute
log_env densemine
if [ "${MODE:-embed}" = "search" ]; then
  python -m ragroute.data.mine_dense search --num-shards 8
else
  python -m ragroute.data.mine_dense embed --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8
fi
