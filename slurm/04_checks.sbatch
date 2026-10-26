#!/bin/bash
#SBATCH --job-name=rr-checks
#SBATCH --account=VERIFY_ACCOUNT
#SBATCH --partition=VERIFY_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Week-1 validity checks (small generator only, 1 GPU). Run after 03_generate has produced qwen1p5b caches.
#  1) staged runner == AutoRAG's own end-to-end pipeline (2 configs, 50 queries)
#  2) tokenizer input-token count == vLLM's prompt_token_ids length
#  3) greedy-decoding reproducibility: re-run one config into a separate folder and diff
source slurm/common.sh
conda activate ragroute
log_env checks
python -m ragroute.harness.check_equivalence --config-id hybrid-k3-rr1-qwen1p5b --n 50
python -m ragroute.harness.check_equivalence --config-id bm25-k10-rr0-qwen1p5b --n 50
python -m ragroute.harness.run_stages check_tokens --generator qwen1p5b
python -m ragroute.harness.run_stages generate --generator qwen1p5b --only dense-k3-rr0-qwen1p5b \
       --subdir generation_repro --force
python - <<'EOF'
import json, os, pandas as pd
d = os.environ["DATA_ROOT"] + "/cache"
a = pd.read_parquet(f"{d}/generation/dense-k3-rr0-qwen1p5b.parquet").set_index("qid")["generated_text"]
b = pd.read_parquet(f"{d}/generation_repro/dense-k3-rr0-qwen1p5b.parquet").set_index("qid")["generated_text"]
rep = {"config": "dense-k3-rr0-qwen1p5b", "n": int(len(a)), "same_text_rate": float((a == b.loc[a.index]).mean())}
print(rep)
open(os.environ["RESULTS_ROOT"] + "/greedy_reproducibility.json", "w").write(json.dumps(rep))
EOF
