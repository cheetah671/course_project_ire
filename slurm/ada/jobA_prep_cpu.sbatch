#!/bin/bash
#SBATCH -J rr-prepA
#SBATCH -A cs4406
#SBATCH -p u22
#SBATCH -c 10
#SBATCH --mem=29G                   # QoS: 10 CPUs x MaxMemPerCPU 3000 MB
#SBATCH --time=06:00:00
#SBATCH -o /home2/%u/ragroute_run/logs/%x_%j.out   # %u = user name (home is /home2/<user>)
# Job A (CPU only): (pilot: build llama.cpp), sample datasets, download DPR Wikipedia + Pyserini BM25 index to the node SSD,
# mine full-corpus BM25 candidates, build the sub-corpus, split, write AutoRAG YAMLs.
# Pilot:  sbatch --export=ALL,RAGROUTE_PILOT=1 slurm/ada/jobA_prep_cpu.sbatch
# Every step skips work already done, so resubmitting after a timeout resumes.
set -euo pipefail
source "$HOME/ragroute_run/code/slurm/ada/env.sh"
cd "$CODE"
log_env prepA

if [ "$RAGROUTE_BACKEND" = "llamacpp" ]; then   # pilot only; the full run generates with vLLM
  step "build llama.cpp"
  bash slurm/ada/build_llamacpp.sh
fi

step "sample datasets"
[ -s "$DATA_ROOT/qa_raw.parquet" ] || py -m ragroute.data.prepare_qa

step "download DPR passages to node SSD (resumable)"
if [ "$(stat -c %s "$DPR_TSV" 2>/dev/null || echo 0)" != "$DPR_BYTES" ]; then
  wget -c -q --tries=30 --waitretry=10 --retry-connrefused -O "$DPR_TSV.part" "$DPR_URL"
  [ "$(stat -c %s "$DPR_TSV.part")" = "$DPR_BYTES" ] && mv "$DPR_TSV.part" "$DPR_TSV"
fi
ls -la "$DPR_TSV"; (zcat "$DPR_TSV" | head -2 | cut -c1-150) || true

# Full-corpus BM25 index. Default (7 Nov 2026): BUILD it from the DPR TSV. The prebuilt Pyserini index (9.2 GB) came
# from its server at ~0.2 MB/s (~12 h); the TSV downloads in minutes and Lucene indexes 21M passages in < 1 h.
# BM25_INDEX=prebuilt restores the pilot's route (wget -c + MD5 + unpack).
if [ "${BM25_INDEX:-build}" = "prebuilt" ]; then
  step "download + verify + unpack the Pyserini BM25 index (resumable; Pyserini's own downloader cannot resume)"
  if ! ls "$IDX_DIR"/*/segments_* >/dev/null 2>&1; then
    mkdir -p "$(dirname "$IDX_TAR")" "$IDX_DIR"
    wget -c -q --tries=30 --waitretry=10 --retry-connrefused -O "$IDX_TAR" "$IDX_URL"
    echo "$IDX_MD5  $IDX_TAR" | md5sum -c -
    tar -xzf "$IDX_TAR" -C "$IDX_DIR" && rm -f "$IDX_TAR"
  fi
  INDEX_PATH=$(dirname "$(ls "$IDX_DIR"/*/segments_* | head -1)")
else
  JSONL=$NODE_SCRATCH/dpr_jsonl
  INDEX_PATH=$NODE_SCRATCH/bm25_index_built
  if [ ! -f "$INDEX_PATH/_DONE" ]; then
    step "DPR TSV -> Pyserini JSONL shards"
    [ -f "$JSONL/_DONE" ] || py -m ragroute.data.dpr_to_jsonl --tsv "$DPR_TSV" --out "$JSONL" --shards "$SLURM_CPUS_PER_TASK"
    step "build the Lucene BM25 index (Pyserini defaults, $SLURM_CPUS_PER_TASK threads)"
    rm -rf "$INDEX_PATH"
    pymine -m pyserini.index.lucene --collection JsonCollection --input "$JSONL" --index "$INDEX_PATH" \
         --generator DefaultLuceneDocumentGenerator --threads "$SLURM_CPUS_PER_TASK"
    date -Is > "$INDEX_PATH/_DONE"
  fi
fi
echo "index at $INDEX_PATH"; du -sh "$INDEX_PATH"

step "BM25 mining over all 21M passages"
[ -s "$DATA_ROOT/mined_bm25.parquet" ] || pymine -m ragroute.data.mine_candidates --threads "$SLURM_CPUS_PER_TASK" \
     --index-dir "$INDEX_PATH"

step "build sub-corpus + AutoRAG corpus/qa parquet"
[ -s "$DATA_ROOT/autorag/corpus.parquet" ] || py -m ragroute.data.build_corpus

step "grouped split + leakage report"
py -m ragroute.data.splits

step "AutoRAG YAMLs"
py -m ragroute.harness.make_autorag_yaml

step "DONE"; du -sh "$DATA_ROOT" "$NODE_SCRATCH"
