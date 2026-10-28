#!/bin/bash
# Source in every Ada job (and interactive shell) for this project:  source ~/ragroute_run/code/slurm/ada/env.sh
# Account cs4406.11 limits (QoS cs4406): 1 GPU, 10 CPUs, 1 running + 1 submitted job, 6 h wall time.
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export MM=$HOME/.local/bin/micromamba
export RUN=$HOME/ragroute_run
export CODE=${CODE:-$RUN/code}

# Pilot (mid-submission) vs full run keep separate data/results folders
if [ "${RAGROUTE_PILOT:-0}" = "1" ]; then
  export DATA_ROOT=$RUN/pilot/data RESULTS_ROOT=$RUN/pilot/results
else
  export DATA_ROOT=$RUN/data RESULTS_ROOT=$RUN/results
fi
export AUTORAG_YAML_DIR=$DATA_ROOT/autorag_yaml   # YAMLs embed the Chroma path, so keep them per data root

# Node-local SSD (fast, large, NOT shared between nodes): big downloads that can be re-fetched
export NODE_SCRATCH=/ssd_scratch/$USER/ragroute
mkdir -p "$NODE_SCRATCH" 2>/dev/null || export NODE_SCRATCH=/tmp/$USER/ragroute   # node without /ssd_scratch
export HF_HOME=$NODE_SCRATCH/hf_cache
export PYSERINI_CACHE=$NODE_SCRATCH/pyserini
export DPR_TSV=$NODE_SCRATCH/raw/psgs_w100.tsv.gz
export DPR_URL=https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
export DPR_BYTES=4694541059
# Pyserini prebuilt Lucene index of the same 21,015,324 DPR passages (URL/MD5/size from Pyserini 1.6.0 metadata)
export IDX_URL=https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz
export IDX_MD5=7b58c08da992b2ea7e96667f0b176651
export IDX_TAR=$NODE_SCRATCH/pyserini_dl/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz
export IDX_DIR=$NODE_SCRATCH/pyserini_idx

# llama.cpp backend (GTX 1080 Ti): binary built by build_llamacpp.sh; CUDA libs passed only to llama-server
export LLAMA_SERVER_BIN=$RUN/llama.cpp/build/bin/llama-server
export LLAMA_LIB_PATH=$MAMBA_ROOT_PREFIX/envs/llamacpp_build/lib

export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$RUN/logs" "$NODE_SCRATCH/raw" "$HF_HOME" "$PYSERINI_CACHE" 2>/dev/null || true

# RAGROUTE_ENV=ragroute-vllm selects the vLLM env (2080 Ti nodes); default is the pilot's llama.cpp-era env
py ()     { "$MM" run -n "${RAGROUTE_ENV:-ragroute}" python "$@"; }
# Pyserini 1.6 builds an OpenAI client at import time and fails without a key; a dummy key is never used.
pymine () { OPENAI_API_KEY=${OPENAI_API_KEY:-not-used} "$MM" run -n ragroute-mining python "$@"; }
step ()   { echo; echo "== $(date -Is) $*"; }

require_vllm_gpu () {  # vLLM needs compute capability >= 7.5; stop early if SLURM placed us on a GTX 1080 Ti
  local cap; cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1), cc $cap"
  awk -v c="$cap" 'BEGIN { exit !(c >= 7.5) }' || { echo "compute capability $cap < 7.5: vLLM cannot run here"; exit 3; }
}

log_env () {  # reproducibility record per job; never allowed to fail the job
  local tag="$1" f="$RESULTS_ROOT/env_${1}_${SLURM_JOB_ID:-login}.txt"
  {
    echo "job=${SLURM_JOB_ID:-none} host=$(hostname) date=$(date -Is) pilot=${RAGROUTE_PILOT:-0}"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null || echo "no GPU"
    "$MM" run -n "${RAGROUTE_ENV:-ragroute}" python -c "import importlib.metadata as m, torch; print('torch', torch.__version__, 'archs', torch._C._cuda_getArchFlags(), 'cuda_ok', torch.cuda.is_available(), '| autorag', m.version('autorag'), '| pandas', m.version('pandas'))" 2>&1 | tail -1
  } > "$f" 2>&1 || true
  "$MM" run -n "${RAGROUTE_ENV:-ragroute}" python -m pip freeze > "${f%.txt}_pip.txt" 2>/dev/null || true
  echo "env logged to $f"
}
