#!/bin/bash
#SBATCH -J rr-judgeJ
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=04:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job J (1 GPU, after job C): LLM judge (RAGAS faithfulness + answer relevancy).
#  1) sample the 150 dev items for human annotation (QA + zsRE) -> annotation/judge_validation_to_annotate.csv
#  2) judge those 150 items                                       -> judge_validation_ragas.parquet
#  3) judge the test subsample (ragas_queries_per_dataset x 24)  -> ragas_scores.parquet
# Whether (3) may be REPORTED depends on the human validation (docs/05): judge_validation score after annotation.
# The judge is served by vLLM (env ragroute-vllm) on port 8001; RAGAS runs in env ragroute-ragas.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
require_vllm_gpu
log_env judgeJ
JUDGE=$(py -c "from ragroute.config import load_experiment as l; print(l()['judge']['model'])")
MAXLEN=$(py -c "from ragroute.config import load_experiment as l; print(l()['judge']['max_model_len'])")

step "sample 150 validation items"
[ -s "$RESULTS_ROOT/judge_validation_items.parquet" ] || py -m ragroute.eval.judge_validation sample

step "start judge server: $JUDGE"
"$MM" run -n ragroute-vllm vllm serve "$JUDGE" --port 8001 --dtype half --max-model-len "$MAXLEN" \
     --gpu-memory-utilization 0.85 --max-num-seqs 32 --seed 0 --served-model-name "$JUDGE" \
     > "$RESULTS_ROOT/vllm_judge_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -sf http://localhost:8001/v1/models >/dev/null && break; sleep 10; done
curl -sf http://localhost:8001/v1/models >/dev/null || { echo "judge server did not start"; tail -50 "$RESULTS_ROOT/vllm_judge_${SLURM_JOB_ID}.log"; exit 4; }

RAGAS="$MM run -n ragroute-ragas python"
step "judge the 150 validation items"
[ -s "$RESULTS_ROOT/judge_validation_ragas.parquet" ] || $RAGAS -m ragroute.eval.ragas_eval \
   --items "$RESULTS_ROOT/judge_validation_items.parquet" --out "$RESULTS_ROOT/judge_validation_ragas.parquet"
step "judge the test subsample (all 24 configs)"
[ -s "$RESULTS_ROOT/ragas_scores.parquet" ] || $RAGAS -m ragroute.eval.ragas_eval
step "DONE"; ls -la "$RESULTS_ROOT"/judge_validation_* "$RESULTS_ROOT"/ragas_scores.parquet
