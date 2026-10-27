#!/bin/bash
# One-time environment setup on the Ada LOGIN node (network-bound installs only; compiling happens in job A).
#   1. ragroute env: replace the cu128 PyTorch (no GTX 1080 Ti / sm_61 kernels) with a cu126 build that has them
#   2. ragroute-mining env: OpenJDK 21 + CPU PyTorch + Pyserini (full-corpus BM25 mining)
#   3. llamacpp_build env: cmake + CUDA 12.6 compiler/libs to build llama.cpp for sm_61 (and sm_75)
# Usage: nohup bash slurm/ada/setup_envs.sh > ~/ragroute_run/logs/setup_envs.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/env.sh"
PIP="--no-cache-dir --progress-bar off"

step "1. ragroute: PyTorch with sm_61 support (cu126)"
"$MM" run -n ragroute python -m pip install $PIP "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
"$MM" run -n ragroute python -c "import torch; a=torch._C._cuda_getArchFlags(); print('torch', torch.__version__, a); assert 'sm_61' in a, 'no sm_61 kernels'"
"$MM" run -n ragroute python -m pip install $PIP --no-deps -e "$CODE"

step "2. ragroute-mining: JDK 21 + CPU torch + pyserini"
"$MM" create -y -n ragroute-mining -c conda-forge python=3.11 openjdk=21
"$MM" run -n ragroute-mining python -m pip install $PIP torch --index-url https://download.pytorch.org/whl/cpu
"$MM" run -n ragroute-mining python -m pip install $PIP pyserini "pandas>=2.2.3,<3" pyarrow pyyaml
"$MM" run -n ragroute-mining python -m pip install $PIP --no-deps -e "$CODE"
"$MM" run -n ragroute-mining python -c "import pyserini, jnius_config; print('pyserini OK')"

step "3. llamacpp_build: cmake + CUDA 12.6 toolchain"
"$MM" create -y -n llamacpp_build -c conda-forge cmake make git "cuda-version=12.6" cuda-nvcc cuda-cudart-dev \
     libcublas-dev cuda-driver-dev "gcc_linux-64=13" "gxx_linux-64=13"
"$MM" run -n llamacpp_build nvcc --version | tail -2

step "disk usage"
du -sh "$MAMBA_ROOT_PREFIX"/envs/* ; df -h "$HOME" | tail -1
step "DONE"
