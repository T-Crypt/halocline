#!/usr/bin/env bash
set -u
O=$(cd "$(dirname "$0")" && pwd)
. "$O/bench.env"
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
power() { sudo "$TUNE" --power "$1" --max-clock 2600 --offset 150 >> "$O/gpu-tune.log" 2>&1; log "power limit now $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"; }
watch_thermal() {   # 1 s samples; at hw slowdown or 83 C while above 350 W, back to 350 W and mark the run
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
    --format=csv,noheader -lms 1000 | while IFS=, read -r ts temp pw pl sm hw hwt swt; do
      echo "$ts,$temp,$pw,$pl,$sm,$hw,$hwt,$swt" >> "$O/thermal-$1.csv"
      t=${temp// /}; l=${pl// /}; l=${l%%.*}
      h1=${hw// /}; h2=${hwt// /}
      if [ "${l:-0}" -gt 350 ] && { [ "${t:-0}" -ge 83 ] || [ "$h1" = Active ] || [ "$h2" = Active ]; }; then
        echo "THERMAL $1 $ts temp=$t hw=$hw hwt=$hwt" >> "$O/run.log"; power 350; touch "$O/thermal-$1.tripped"
      fi
    done
}
run() {   # name pack watts
  [ "$3" = 450 ] && power 450
  watch_thermal "$1" & W=$!
  log "$1 bench start: $(basename "$2") at $3 W"
  "$R/bin/halocline-bench" --community --model "$2" --submitter T-Crypt --community-file "$WT/docs/community/benchmarks.json" \
      > "$O/$1.bench.txt" 2>&1
  log "$1 bench rc=$? $( [ -f "$O/thermal-$1.tripped" ] && echo THERMAL-LIMITED )"
  kill $W 2>/dev/null; pkill -P $W 2>/dev/null; pkill -f "nvidia-smi --query-gpu=timestamp" 2>/dev/null
  [ "$3" = 450 ] && power 350
}
run A-ud-350 "$UD" 350
run B-sc117-350 "$PACK" 350
run C-sc117-450 "$PACK" 450
run D-ud-450 "$UD" 450
sudo systemctl restart gpu-tune.service; log "gpu-tune.service restarted: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
