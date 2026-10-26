#!/bin/bash
# Run ONCE on an Ada login node (needs internet). Creates the three conda envs and prefetches models/data.
# Usage: bash slurm/00_setup_env.sh
set -euo pipefail
source slurm/common.sh || true   # for paths only; set -e is re-enabled below
set -euo pipefail
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

conda create -y -n ragroute python=3.11
conda activate ragroute
pip install -r requirements-cluster.txt
pip check || echo "WARNING: pip check reported conflicts - resolve before the pilot"

conda create -y -n ragroute-mining python=3.11 openjdk=21 -c conda-forge   # VERIFY JDK version for your Pyserini
conda activate ragroute-mining
pip install -r requirements-mining.txt

conda create -y -n ragroute-ragas python=3.11
conda activate ragroute-ragas
pip install -r requirements-ragas.txt

# Prefetch models into $HF_HOME (compute nodes are assumed offline)
conda activate ragroute
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "BAAI/bge-base-en-v1.5",
          "cross-encoder/ms-marco-MiniLM-L-6-v2"]:
    print(m, snapshot_download(m))
EOF
# Datasets (cached into $HF_HOME/datasets)
python -m ragroute.data.prepare_qa
# Wikipedia passages (VERIFY URL and size before downloading ~13 GB uncompressed)
mkdir -p "$DATA_ROOT/raw"
wget -c -O "$DATA_ROOT/raw/psgs_w100.tsv.gz" https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip -k "$DATA_ROOT/raw/psgs_w100.tsv.gz"
head -3 "$DATA_ROOT/raw/psgs_w100.tsv"
wc -l "$DATA_ROOT/raw/psgs_w100.tsv"   # expect 21,015,325 lines incl. header (VERIFY)
# Pyserini prebuilt index download (large; VERIFY name)
conda activate ragroute-mining
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')"
