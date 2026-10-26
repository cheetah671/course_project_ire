#!/bin/bash
#SBATCH --job-name=rr-judge
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# RAGAS judge. MODE=validate scores the 150 hand-labelled items; MODE=full scores the test subsample
# (only after validation passes the pre-registered threshold, see docs/03_judge_annotation_guide.md).
# The judge runs as a vLLM OpenAI-compatible server in the ragroute env; RAGAS runs in its own env.
source slurm/common.sh
conda activate ragroute
JUDGE_MODEL=$(python -c "import yaml;print(yaml.safe_load(open('configs/experiment.yaml'))['judge']['model'])")
log_env judge
vllm serve "$JUDGE_MODEL" --port 8001 --tensor-parallel-size 2 --dtype half --max-model-len 8192 \
     --seed 0 > "logs/vllm_judge_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID' EXIT
until curl -sf http://localhost:8001/v1/models > /dev/null; do sleep 10; done

conda activate ragroute-ragas
if [ "${MODE:-validate}" = "validate" ]; then
  python -m ragroute.eval.ragas_eval --items "$RESULTS_ROOT/judge_validation_items.parquet" \
         --out "$RESULTS_ROOT/judge_validation_ragas.parquet"
else
  python -m ragroute.eval.ragas_eval
fi
