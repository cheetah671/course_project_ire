#!/bin/bash
# Submits the whole pipeline with SLURM dependencies. Run from the repo root on an Ada login node,
# after slurm/00_setup_env.sh. For the PILOT, first set corpus.subcorpus_size: 100000 and datasets.*.n: 100.
set -euo pipefail
mkdir -p logs
j1=$(sbatch --parsable slurm/01_build_corpus.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/02_retrieval_rerank.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j2 slurm/03_generate.sbatch)
j5=$(sbatch --parsable --dependency=afterok:$j3 slurm/05_collect_router.sbatch)
echo "submitted: corpus=$j1 retrieval=$j2 generate=$j3 router=$j5"
echo "then (manual): 04_checks (week 1), judge validation (sample -> annotate -> 07_judge MODE=validate), 06_latency"
