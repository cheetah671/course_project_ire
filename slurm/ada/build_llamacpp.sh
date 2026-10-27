#!/bin/bash
# Build llama.cpp's llama-server with CUDA for GTX 1080 Ti (sm_61) and RTX 2080 Ti (sm_75). Run inside job A
# (needs ~10 CPUs for ~10-15 min; no GPU needed to compile). Pinned release for reproducibility.
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG=${LLAMACPP_TAG:-b11936}
if [ -x "$LLAMA_SERVER_BIN" ]; then echo "llama-server already built: $LLAMA_SERVER_BIN"; exit 0; fi
cd "$RUN"
[ -d llama.cpp ] || git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp
"$MM" run -n llamacpp_build bash -c "
  set -e; cd $RUN/llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='61;75' -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j ${SLURM_CPUS_PER_TASK:-8} --target llama-server
"
ls -la "$LLAMA_SERVER_BIN"
LD_LIBRARY_PATH=$LLAMA_LIB_PATH "$LLAMA_SERVER_BIN" --version 2>&1 | tail -2 || true
