#!/bin/bash
# Unattended driver for the rest of the full run, on the Ada LOGIN node (the QoS allows only ONE job in the queue,
# so jobs cannot be chained with --dependency; this loop submits each job after the previous one has finished).
# For each step: submit, wait, check its success marker, resubmit after a time-out (up to a limit), then go on.
# Between B1 and K it sets the 1.5B generator's dtype from the fp16-vs-fp32 check (rule in decide_dtype).
#
# Order: B1 -> K -> B2 -> K2 -> D -> C -> J.
# Usage: nohup bash ~/ragroute_run/code/slurm/ada/run_all_driver.sh [FIRST_STEP] [RUNNING_JOBID] \
#              > ~/ragroute_run/logs/driver.log 2>&1 < /dev/null &
#   FIRST_STEP: B1 | K | B2 | K2 | D | C | J (default B1). RUNNING_JOBID: an already-submitted job of FIRST_STEP to wait for.
# Progress: tail ~/ragroute_run/logs/driver.log. Stop: kill the driver's PID (the running job is not touched).
set -uo pipefail
RUN=$HOME/ragroute_run
CODE=$RUN/code
ADA=$CODE/slurm/ada
RES=$RUN/results
log () { echo "[$(date '+%F %T')] $*"; }

declare -A SCRIPT=([B1]=jobB1_retrieval_gpu.sbatch [K]=jobK_calibrate_gpu.sbatch [B2]=jobB2_generate_gpu.sbatch
                   [K2]=jobK2_index_cost_gpu.sbatch [D]=jobD_checks_gpu.sbatch [C]=jobC_analysis_cpu.sbatch [J]=jobJ_judge_gpu.sbatch)
declare -A NAME=([B1]=rr-retrB1 [K]=rr-calibrate [B2]=rr-genB2 [K2]=rr-indexK2 [D]=rr-checksD [C]=rr-analysisC [J]=rr-judgeJ)
declare -A TRIES=([B1]=2 [K]=2 [B2]=4 [K2]=2 [D]=2 [C]=5 [J]=3)
ORDER=(B1 K B2 K2 D C J)

done_marker () {  # success = the step's own completion marker exists
  case $1 in
    B1) [ -s "$RES/dtype_check_qwen1p5b.json" ] ;;
    K)  [ -s "$CODE/configs/cost_weights_measured.yaml" ] ;;
    B2) [ -s "$RES/runs_scored.parquet" ] && [ -s "$RES/offline_cost.json" ] ;;
    K2) [ -s "$RES/index_calibration.json" ] ;;
    D)  [ -s "$RES/greedy_reproducibility.json" ] ;;
    C)  ls "$RES"/figures/F2_pareto_test.png >/dev/null 2>&1 && [ -s "$RES/tables/T4_T5_router_headline_and_ablations.csv" ] ;;
    J)  [ -s "$RES/ragas_scores.parquet" ] ;;
  esac
}

wait_job () {  # returns only when sacct reports a final state (a failing squeue/sacct call never counts as "finished")
  local j=$1 st
  while true; do
    st=$(sacct -j "$j" -X -n -o State 2>/dev/null | head -1 | xargs)
    case "$st" in
      COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*) break ;;
    esac
    sleep 120
  done
  log "job $j finished: $(sacct -j "$j" -X -n -o State,Elapsed 2>/dev/null | xargs)"
}

submit () {  # retries while the one-job QoS slot is busy or sbatch hiccups
  local step=$1 out j
  while true; do
    out=$(cd "$RUN" && sbatch "$ADA/${SCRIPT[$step]}" 2>&1)
    j=$(echo "$out" | awk '/Submitted batch job/ {print $4}')
    [ -n "$j" ] && { echo "$j"; return 0; }
    log "sbatch $step failed ($out); retrying in 5 min" >&2
    sleep 300
  done
}

decide_dtype () {  # fp32 only if fp16 measurably hurts the 1.5B model (>= 0.01 mean quality on the same 1,000 questions)
  "$HOME/.local/bin/micromamba" run -n ragroute-vllm python - "$RES/dtype_check_qwen1p5b.json" "$CODE/configs/experiment.yaml" "$RES/dtype_decision.json" <<'EOF'
import json, re, sys
rows = json.load(open(sys.argv[1]))
r = rows[0]
q32 = r["quality_generation_dtype_float32"]; q16 = r["quality_generation_dtype_float16"]
choice = "float32" if q32 - q16 >= 0.01 else "float16"
cfg = open(sys.argv[2]).read()
cfg = re.sub(r"(\n\s+dtype: )null(\s+# DTYPE_1P5B)", r"\g<1>" + choice + r"\g<2>", cfg)
open(sys.argv[2], "w").write(cfg)
dec = {"quality_fp32": q32, "quality_fp16": q16, "same_text_rate": r["same_text_rate"],
       "out_tok_fp32": r.get("output_tokens_mean_generation_dtype_float32"),
       "out_tok_fp16": r.get("output_tokens_mean_generation_dtype_float16"),
       "rule": "float32 if quality_fp32 - quality_fp16 >= 0.01 else float16", "choice": choice}
json.dump(dec, open(sys.argv[3], "w"), indent=2)
print(json.dumps(dec))
EOF
}

FIRST=${1:-B1}
RUNNING=${2:-}
started=0
for step in "${ORDER[@]}"; do
  [ "$step" = "$FIRST" ] && started=1
  [ $started -eq 1 ] || continue
  if done_marker "$step"; then log "$step already complete; skipping"; else
    for try in $(seq 1 "${TRIES[$step]}"); do
      if [ -n "$RUNNING" ]; then j=$RUNNING; RUNNING=""; log "$step: waiting for already-running job $j"
      else j=$(submit "$step"); log "$step: submitted job $j (try $try)"; fi
      wait_job "$j"
      done_marker "$step" && break
      log "$step: completion marker missing after job $j; last log lines:"
      tail -5 "$RUN/logs/${NAME[$step]}_$j.out" 2>/dev/null | sed 's/^/    /'
    done
    if ! done_marker "$step"; then log "STOP: $step did not complete after ${TRIES[$step]} tries"; exit 1; fi
  fi
  if [ "$step" = "B1" ] && grep -q "dtype: null" "$CODE/configs/experiment.yaml"; then
    log "precision decision for qwen1p5b: $(decide_dtype)"
  fi
done
log "ALL STEPS COMPLETE"
