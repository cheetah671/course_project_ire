#!/bin/bash
# One-time setup on the Ada LOGIN node: env "ragroute-ragas" (RAGAS + LangChain), kept apart because RAGAS's
# LangChain pins conflict with AutoRAG's. The judge LLM itself is served by vLLM from env "ragroute-vllm";
# this env only sends HTTP requests to it and embeds answers for answer relevancy (CPU torch is enough).
# Usage: bash ~/ragroute_run/code/slurm/ada/setup_ragas_env.sh > ~/ragroute_run/logs/setup_ragas.log 2>&1
set -euo pipefail
source "$(dirname "$0")/env.sh"
ENV=ragroute-ragas
[ -x "$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python" ] || "$MM" create -y -n $ENV -c conda-forge python=3.11
PY=$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python
cd "$CODE"   # requirements-ragas.txt installs the project with a relative "-e ."
UV=$RUN/bin/uv
"$UV" pip install --python "$PY" --no-cache torch --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$PY" --no-cache -r "$CODE/requirements-ragas.txt"
"$MM" run -n $ENV python -c "import ragas, langchain_openai, langchain_huggingface; print('ragas', ragas.__version__)"
"$MM" clean -a -y >/dev/null || true
du -sh "$MAMBA_ROOT_PREFIX/envs/$ENV"; df -h "$HOME" | tail -1
step "DONE"
