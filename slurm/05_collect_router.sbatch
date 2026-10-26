#!/bin/bash
#SBATCH --job-name=rr-router
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# CPU only. Scores the grid, builds the oracle table, trains/evaluates all routers and baselines,
# writes tables and figures. Also re-runs the main routers under alternative cost-weight sets.
source slurm/common.sh
conda activate ragroute
log_env router
python -m ragroute.harness.collect
python -m ragroute.router.train_eval
python -m ragroute.router.train_eval --models gbdt --feature-sets B --no-reuse   # conservative Router B accounting
python -m ragroute.analysis.tables
python -m ragroute.analysis.plots
