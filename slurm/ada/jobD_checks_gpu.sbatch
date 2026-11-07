#!/bin/bash
#SBATCH -J rr-checksD
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode[001-008,011-012,017,020-023,025-029,031,033-036,038,040-041]  # = all u22 nodes below gnode042
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job D (1 GPU, after job B): validity checks on the small generator.
#  1) staged runner == AutoRAG's own end-to-end pipeline (2 configs, 50 queries each)
#  2) our input-token count == the serving engine's prompt length (also checks the chat template)
#  3) greedy-decoding reproducibility: re-run one config into a separate folder and compare texts
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobD_checks_gpu.sbatch
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
log_env checksD

[ "$RAGROUTE_BACKEND" = "vllm" ] && require_vllm_gpu
step "regenerate AutoRAG YAMLs (per-generator dtype is fixed only after job B1)"
py -m ragroute.harness.make_autorag_yaml
step "token count check"
[ -s "$RESULTS_ROOT/token_count_check_qwen1p5b.json" ] || py -m ragroute.harness.run_stages check_tokens --generator qwen1p5b
step "equivalence: hybrid-k3-rr1-qwen1p5b"
py -m ragroute.harness.check_equivalence --config-id hybrid-k3-rr1-qwen1p5b --n 50
step "equivalence: bm25-k10-rr0-qwen1p5b"
py -m ragroute.harness.check_equivalence --config-id bm25-k10-rr0-qwen1p5b --n 50
step "greedy reproducibility: dense-k3-rr0-qwen1p5b"
py -m ragroute.harness.run_stages generate --generator qwen1p5b --only dense-k3-rr0-qwen1p5b \
   --subdir generation_repro --force
REPRO=$(cat <<'EOF'
import json, os, pandas as pd
d = os.environ["DATA_ROOT"] + "/cache"
a = pd.read_parquet(f"{d}/generation/dense-k3-rr0-qwen1p5b.parquet").set_index("qid")["generated_text"]
b = pd.read_parquet(f"{d}/generation_repro/dense-k3-rr0-qwen1p5b.parquet").set_index("qid")["generated_text"]
rep = {"config": "dense-k3-rr0-qwen1p5b", "n": int(len(a)), "same_text_rate": float((a == b.loc[a.index]).mean())}
print(rep)
open(os.environ["RESULTS_ROOT"] + "/greedy_reproducibility.json", "w").write(json.dumps(rep))
EOF
)
py -c "$REPRO"   # micromamba run may not forward stdin, so pass the script as an argument
step "DONE"; cat "$RESULTS_ROOT"/equivalence_*.json "$RESULTS_ROOT"/token_count_check_*.json "$RESULTS_ROOT"/greedy_reproducibility.json
