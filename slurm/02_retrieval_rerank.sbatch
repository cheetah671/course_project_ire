#!/bin/bash
#SBATCH --job-name=rr-retrieve
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# AutoRAG retrieval trial (BM25 + Chroma ingestion once, then bm25/dense/hybrid top-20 for every query),
# then cross-encoder reranking of each pool once. Both are cached; re-submitting skips finished work.
# Expected bottleneck: rank_bm25 over ~1M passages (seconds/query, CPU). Pilot with subcorpus_size=100000 first.
source slurm/common.sh
conda activate ragroute
log_env retrieval
python -m ragroute.harness.run_stages retrieval
python -m ragroute.harness.run_stages rerank
python -m ragroute.features.build_features --device cuda
python scripts/record_offline_cost.py
