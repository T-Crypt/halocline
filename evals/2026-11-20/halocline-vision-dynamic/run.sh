#!/usr/bin/env bash
# Sparse image encoder: --vision cpu (served) vs --vision dynamic (in the engine, on the 4090), 2026-11-20.
# Why: the 2026-11-19 airbench QFN run spent ~340 s of a 15m41s vision section waiting for the CPU encoder
# (median TTFT 10.6 s on image turns, 0.7 s on text turns).
# Run under: berth train --as halocline-vision-ab -- ./run.sh [ARM ...]
# Arms (release aa156951, the served sparse flags, :8096); only --vision differs:
#   cpu      --vision cpu        (served today)
#   dynamic  --vision dynamic
# Default order ABBA: cpu dynamic dynamic cpu. Each arm: known-answer vision_check (3 images), vision_timing.py over the
# 19 airbench vision images (outside the repo, /var/tmp/vision-ab), then lsw.py smoke --long (speed, needle, 137K).
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
R=$HOME/halocline/releases/aa156951
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
PACK=/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
AB=/var/tmp/vision-ab
export TMPDIR=/var/tmp/halocline
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }

exec 9>/tmp/halocline-build.lock
flock 9
log "build lock held; load $(cut -d" " -f1-3 /proc/loadavg); power: $(nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader)"

n=0
for arm in ${*:-cpu dynamic dynamic cpu}; do
  n=$((n + 1)); tag=$n-$arm
  case $arm in
    cpu|dynamic) extra="--vision $arm" ;;
    *) log "unknown arm $arm"; continue ;;
  esac
  log "$tag start: $extra; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  "$R/bin/halocline" serve "$PACK" --python "$PY" --host 127.0.0.1 --port 8096 --alias vision-ab \
      --spec 8 --prefill auto --max-context 262144 --kv int8 $extra --ram-reserve-mib 2048 > "$O/$tag.serve.out" 2>&1 &
  SRV=$!
  ( while kill -0 $SRV 2>/dev/null; do
      echo "$(date +%T) avail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
      sleep 2; done ) > "$O/$tag.mem.log" 2>&1 &
  TR=$!
  for t in $(seq 1 900); do grep -q "weights  loaded" "$O/$tag.serve.out" && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  for t in $(seq 1 120); do curl -sf -m 2 http://127.0.0.1:8096/v1/models >/dev/null && break; sleep 1; done
  log "$tag ready: $(grep -m1 -oE 'vision=[a-z0-9]+' "$O/$tag.serve.out")"
  python3 "$VC" vision-ab --url http://127.0.0.1:8096 --out "$O/$tag.check.json" > "$O/$tag.check.txt" 2>&1
  log "$tag vision_check rc=$?"; sed "s/^/$tag  /" "$O/$tag.check.txt" | tee -a "$O/run.log"
  python3 "$O/vision_timing.py" vision-ab --url http://127.0.0.1:8096 --images $AB/img --prompts $AB/s2_vision.json \
      --out "$O/$tag.timing.json" > "$O/$tag.timing.txt" 2>&1
  log "$tag timing rc=$?"; grep -E '^(summary|text probe)' "$O/$tag.timing.txt" | sed "s/^/$tag  /" | tee -a "$O/run.log"
  python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke vision-ab --url http://127.0.0.1:8096 --long \
      --out "$O/$tag.smoke.json" > "$O/$tag.smoke.txt" 2>&1
  log "$tag smoke rc=$?"; grep -E '^(PASS|FAIL)' "$O/$tag.smoke.txt" | sed "s/^/$tag  /" | tee -a "$O/run.log"
  kill -TERM $SRV; wait $SRV 2>/dev/null; kill $TR 2>/dev/null
  sleep 5
done
log "done"
