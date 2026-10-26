#!/bin/bash
#SBATCH --job-name=rr-corpus
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only: BM25 mining over the full corpus (Pyserini), sub-corpus build, splits, AutoRAG YAMLs.
source slurm/common.sh
conda activate ragroute-mining
log_env mining
python -m ragroute.data.mine_candidates --threads "$SLURM_CPUS_PER_TASK"

conda activate ragroute
log_env corpus
python -m ragroute.data.build_corpus
python -m ragroute.data.splits
python -m ragroute.harness.make_autorag_yaml
