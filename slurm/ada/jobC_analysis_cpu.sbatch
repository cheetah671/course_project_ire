#!/bin/bash
#SBATCH -J rr-analysisC
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=03:00:00
#SBATCH -o /home2/cs4406.11/ragroute_run/logs/%x_%j.out
# Job C (CPU only, after job B): routers + baselines + oracle, then tables and figures.
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobC_analysis_cpu.sbatch
# The MLP router is skipped here (first in the cut order); add it for the full run if time allows.
set -euo pipefail
source /home2/cs4406.11/ragroute_run/code/slurm/ada/env.sh
cd "$CODE"
log_env analysisC

step "collect + score the grid (no-op if job B already wrote it)"
[ -s "$RESULTS_ROOT/runs_scored.parquet" ] || py -m ragroute.harness.collect
step "routers + baselines (LR, GBDT; all feature sets; 5 seeds)"
py -m ragroute.router.train_eval --models lr gbdt
step "conservative Router B accounting (probe never reused)"
py -m ragroute.router.train_eval --models gbdt --feature-sets B --no-reuse
step "tables"
py -m ragroute.analysis.tables
step "figures"
py -m ragroute.analysis.plots
step "DONE"; ls -la "$RESULTS_ROOT" "$RESULTS_ROOT"/tables "$RESULTS_ROOT"/figures 2>/dev/null || true
