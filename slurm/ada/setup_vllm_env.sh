#!/bin/bash
# One-time setup on the Ada LOGIN node: env "ragroute-vllm" (AutoRAG + vLLM) for the RTX 2080 Ti nodes (gnode042+).
# Separate from env "ragroute" so the pilot's environment (torch cu126 with GTX 1080 Ti kernels) stays unchanged.
# Usage: nohup bash slurm/ada/setup_vllm_env.sh > ~/ragroute_run/logs/setup_vllm_env.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/env.sh"
PIP="--no-cache-dir --progress-bar off"
ENV=ragroute-vllm

step "create $ENV (Python 3.11: AutoRAG 0.3.27 needs <3.13)"
[ -x "$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python" ] || "$MM" create -y -n $ENV -c conda-forge python=3.11
PY=$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python

# uv instead of pip: on the login node pip downloaded at ~100 KB/s (curl from the same host: ~6 MB/s), which made
# the ~3 GB of wheels an hours-long install (2026-10-28). uv downloads in parallel and was fast.
UV=$RUN/bin/uv
if [ ! -x "$UV" ]; then  # uv's PyPI wheel via curl (GitHub release downloads were also slow from the login node)
  mkdir -p "$RUN/bin"; WHL=$(mktemp -d)/uv.whl
  URL=$(curl -s https://pypi.org/pypi/uv/json | "$PY" -c 'import json, sys
print(next(u["url"] for u in json.load(sys.stdin)["urls"] if "manylinux" in u["filename"] and "x86_64" in u["filename"]))')
  curl -sSfL -o "$WHL" "$URL"
  "$PY" -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); n = next(n for n in z.namelist() if n.endswith(".data/scripts/uv")); open(sys.argv[2], "wb").write(z.read(n))' "$WHL" "$UV"
  chmod +x "$UV"
fi
"$UV" --version

# vLLM 0.11.x: built on torch 2.8 + CUDA 12.8, which Ada's driver 570 supports. Newer vLLM pulls torch on CUDA 13
# (needs driver >= 580). AutoRAG without its [gpu] extra: that extra pins xgrammar>=0.1.32, vLLM 0.11 pins 0.1.25;
# we only need the two [gpu] packages below (reranker, dense embeddings), xgrammar is for structured output.
step "vLLM 0.11 + AutoRAG"
"$UV" pip install --python "$PY" --no-cache "vllm${VLLM_SPEC:-~=0.11.0}" "autorag==0.3.27" \
     "sentence-transformers>=4.1.0" "llama-index-embeddings-huggingface>=0.5.4" "sentencepiece>=0.2.0" \
     "pandas>=2.2.3,<3" "datasets>=3.5.1" "pyarrow>=23.0.1" "pytest>=8" pip
"$UV" pip install --python "$PY" --no-cache --no-deps -e "$CODE"

step "checks"
"$MM" run -n $ENV python -c "
import importlib.metadata as m, torch, vllm, pandas
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch._C._cuda_getArchFlags())
print('vllm', vllm.__version__, '| autorag', m.version('autorag'), '| pandas', pandas.__version__)
assert 'sm_75' in torch._C._cuda_getArchFlags(), 'no Turing (sm_75) kernels'
from autorag.nodes.generator import Vllm; import ragroute.harness.run_stages; print('imports OK')"
"$MM" run -n $ENV python -m pip check || true
du -sh "$MAMBA_ROOT_PREFIX/envs/$ENV"; df -h "$HOME" | tail -1
step "DONE"
