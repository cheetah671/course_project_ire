#!/bin/bash
# One-time setup of a FRESH Ada account on the LOGIN node (network-bound installs only, no compiling).
# Installs micromamba, the vLLM env "ragroute-vllm" (harness + generation + analysis) and the Pyserini env
# "ragroute-mining" (full-corpus BM25 mining). The llama.cpp toolchain of the pilot is not installed: the full run
# uses vLLM on the RTX 2080 Ti nodes, and the home quota (25 GB) has no room for a third env.
# Code must already be at ~/ragroute_run/code.
# Usage: nohup bash ~/ragroute_run/code/slurm/ada/bootstrap_account.sh > ~/ragroute_run/logs/bootstrap.log 2>&1 &
set -euo pipefail
mkdir -p "$HOME/.local/bin" "$HOME/ragroute_run/logs"
if [ ! -x "$HOME/.local/bin/micromamba" ]; then
  echo "== install micromamba"
  curl -sSfL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
fi
source "$(dirname "$0")/env.sh"
"$MM" --version

bash "$(dirname "$0")/setup_vllm_env.sh"

step "ragroute-mining: JDK 21 + CPU torch + pyserini"
[ -x "$MAMBA_ROOT_PREFIX/envs/ragroute-mining/bin/python" ] || \
  "$MM" create -y -n ragroute-mining -c conda-forge python=3.11 openjdk=21
MPY=$MAMBA_ROOT_PREFIX/envs/ragroute-mining/bin/python
"$RUN/bin/uv" pip install --python "$MPY" --no-cache torch --index-url https://download.pytorch.org/whl/cpu
"$RUN/bin/uv" pip install --python "$MPY" --no-cache pyserini "pandas>=2.2.3,<3" pyarrow pyyaml
"$RUN/bin/uv" pip install --python "$MPY" --no-cache --no-deps -e "$CODE"
OPENAI_API_KEY=not-used "$MM" run -n ragroute-mining python -c "import pyserini, jnius_config; print('pyserini OK')"

step "clean package caches"
"$MM" clean -a -y >/dev/null || true
du -sh "$MAMBA_ROOT_PREFIX"/envs/* ; df -h "$HOME" | tail -1
step "BOOTSTRAP DONE"
