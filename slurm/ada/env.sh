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
export HF_HOME=$NODE_SCRATCH/hf_cache
export PYSERINI_CACHE=$NODE_SCRATCH/pyserini
export DPR_TSV=$NODE_SCRATCH/raw/psgs_w100.tsv.gz
export DPR_URL=https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

# llama.cpp backend (GTX 1080 Ti): binary built by build_llamacpp.sh; CUDA libs passed only to llama-server
export LLAMA_SERVER_BIN=$RUN/llama.cpp/build/bin/llama-server
export LLAMA_LIB_PATH=$MAMBA_ROOT_PREFIX/envs/llamacpp_build/lib

export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$RUN/logs" "$NODE_SCRATCH/raw" "$HF_HOME" "$PYSERINI_CACHE" 2>/dev/null || true

py ()     { "$MM" run -n ragroute python "$@"; }
pymine () { "$MM" run -n ragroute-mining python "$@"; }
step ()   { echo; echo "== $(date -Is) $*"; }

log_env () {  # reproducibility record per job
  local tag="$1" f="$RESULTS_ROOT/env_${1}_${SLURM_JOB_ID:-login}.txt"
  {
    echo "job=${SLURM_JOB_ID:-none} host=$(hostname) date=$(date -Is) pilot=${RAGROUTE_PILOT:-0}"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null || echo "no GPU"
    "$MM" run -n ragroute python -c "import torch, autorag, pandas; print('torch', torch.__version__, 'archs', torch._C._cuda_getArchFlags(), '| autorag', autorag.__version__, '| pandas', pandas.__version__)" 2>/dev/null
  } > "$f"
  "$MM" run -n ragroute python -m pip freeze > "${f%.txt}_pip.txt" 2>/dev/null
}
