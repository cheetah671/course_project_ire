#!/bin/bash
#SBATCH -J rr-analysisC
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode012          # job 65455: the env silently did not run there
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job C (CPU only, after job B): routers + baselines + oracle, then tables and figures.
# Router runs cache each finished router group and skip themselves once headline.csv exists, so a job stopped
# at 6 h resumes when resubmitted (the full set needs about two jobs at 5,000 questions).
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobC_analysis_cpu.sbatch
# MLP is first in the cut order: MODELS='lr gbdt' sbatch ... drops it.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
log_env analysisC
require_py

step "collect + score the grid (no-op if job B already wrote it)"
if [ ! -s "$RESULTS_ROOT/runs_scored.parquet" ]; then py -m ragroute.harness.collect; fi
step "unit tests (incl. the Chroma-dependent ones)"
if [ ! -f "$RESULTS_ROOT/unit_tests_passed" ]; then   # once per run: retries of job C skip the ~7 min suite
  if py -m pytest -q tests; then date -Is > "$RESULTS_ROOT/unit_tests_passed"; else echo "WARNING: unit tests failed (see above)"; fi
fi
step "routers + baselines (LR, GBDT, MLP; all feature sets; 5 seeds; global + per task)"
if [ ! -s "$RESULTS_ROOT/router/headline.csv" ]; then
  # ~420 router fits take longer than one 6 h job; train_eval caches every finished router group
  # (router/cache/<fingerprint>/), so the driver's resubmission continues where the previous job stopped.
  py -m ragroute.router.train_eval --models ${MODELS:-lr gbdt mlp}
fi
step "conservative Router B accounting (probe never reused)"
if [ ! -s "$RESULTS_ROOT/router_noreuse/headline.csv" ]; then
  py -m ragroute.router.train_eval --models gbdt --feature-sets A B --no-reuse --no-per-task
fi
for W in tokens_only params_scaled api_price_like measured_autorag_index; do
  step "router sensitivity to cost weights: $W"
  if [ ! -s "$RESULTS_ROOT/router_w-$W/headline.csv" ]; then
    py -m ragroute.router.train_eval --models gbdt --feature-sets A B --no-per-task --weight-set $W
  fi
done
step "tables"
py -m ragroute.analysis.tables
step "figures"
py -m ragroute.analysis.plots
step "DONE"; ls -la "$RESULTS_ROOT" "$RESULTS_ROOT"/tables "$RESULTS_ROOT"/figures 2>/dev/null || true
