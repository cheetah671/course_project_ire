#!/bin/bash
#SBATCH -J rr-analysisC
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job C (CPU only, after job B): routers + baselines + oracle, then tables and figures.
# Each router run skips itself once its headline.csv exists, so a job stopped at 6 h resumes when resubmitted
# (~3-4 h in all at 5,000 questions: measured on synthetic data at full scale, 7 Nov).
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobC_analysis_cpu.sbatch
# MLP is first in the cut order: MODELS='lr gbdt' sbatch ... drops it.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
log_env analysisC

step "collect + score the grid (no-op if job B already wrote it)"
[ -s "$RESULTS_ROOT/runs_scored.parquet" ] || py -m ragroute.harness.collect
step "unit tests (incl. the Chroma-dependent ones)"
py -m pytest -q tests || echo "WARNING: unit tests failed (see above)"
step "routers + baselines (LR, GBDT, MLP; all feature sets; 5 seeds; global + per task)"
[ -s "$RESULTS_ROOT/router/headline.csv" ] || py -m ragroute.router.train_eval --models ${MODELS:-lr gbdt mlp}
step "conservative Router B accounting (probe never reused)"
[ -s "$RESULTS_ROOT/router_noreuse/headline.csv" ] || py -m ragroute.router.train_eval --models gbdt --feature-sets A B --no-reuse --no-per-task
for W in tokens_only params_scaled api_price_like; do
  step "router sensitivity to cost weights: $W"
  [ -s "$RESULTS_ROOT/router_w-$W/headline.csv" ] || py -m ragroute.router.train_eval --models gbdt --feature-sets A B --no-per-task --weight-set $W
done
step "tables"
py -m ragroute.analysis.tables
step "figures"
py -m ragroute.analysis.plots
step "DONE"; ls -la "$RESULTS_ROOT" "$RESULTS_ROOT"/tables "$RESULTS_ROOT"/figures 2>/dev/null || true
