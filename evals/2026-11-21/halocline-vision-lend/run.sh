#!/usr/bin/env bash
# Dense image encoder in the borrowed prompt workspace (pre_arena), 2026-11-21, ticket 5KWJVRF.
# Why: dense serves images on the CPU encoder (13-23 s before the first byte per screenshot, live llama-swap captures
# 2026-11-21); --vision dynamic refused every image at 262K because its own 958-1165 MiB arena did not fit.
# Run under: berth train --as halocline-vision-lend -- ./run.sh [ARM ...]
# Arms (the served dense flags, :8097); only the build and --vision differ:
#   cpu   release b2f81178, --vision cpu      (served today)
#   lend  branch vision-pre-arena,  --vision dynamic (encoder borrows pre_arena; own arena, then CPU, as fallbacks)
# Default ABBA: cpu lend lend cpu. Each arm: vision_check (3 known-answer images), vision_timing.py over the 19 airbench
# images (/var/tmp/vision-ab), lsw.py smoke --long (speed, needle, 137K). VRAM sampled every 0.5 s for the peak.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
OLD=$HOME/halocline/releases/b2f81178
NEW=$HOME/wt/halocline-vision-arena
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
AB=/var/tmp/vision-ab
RAW=$AB/raw-2026-11-21-lend; mkdir -p "$RAW"
export TMPDIR=/var/tmp/halocline
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }

exec 9>/tmp/halocline-build.lock
flock 9
log "build lock held; load $(cut -d" " -f1-3 /proc/loadavg); power: $(nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader); new HEAD $(git -C $NEW rev-parse --short HEAD) dirty=$(git -C $NEW status --porcelain -uno | wc -l)"

n=0
for arm in ${*:-cpu lend lend cpu}; do
  n=$((n + 1)); tag=$n-$arm
  case $arm in
    cpu)  R=$OLD; extra="--vision cpu" ;;
    lend) R=$NEW; extra="--vision dynamic" ;;
    *) log "unknown arm $arm"; continue ;;
  esac
  log "$tag start: $R $extra; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
  "$R/bin/halocline" serve "$PACK" --python "$PY" --host 127.0.0.1 --port 8097 --alias vision-ab \
      --spec 8 --long-context $extra --draft-vocab $OLD/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin > "$O/$tag.serve.out" 2>&1 &
  SRV=$!
  ( while kill -0 $SRV 2>/dev/null; do
      echo "$(date +%T.%N | cut -c1-12) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
      sleep 0.5; done ) > "$O/$tag.mem.log" 2>&1 &
  TR=$!
  for t in $(seq 1 900); do grep -q "weights  loaded" "$O/$tag.serve.out" && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  for t in $(seq 1 120); do curl -sf -m 2 http://127.0.0.1:8097/v1/models >/dev/null && break; sleep 1; done
  log "$tag ready: $(grep -m1 -oE 'vision=[a-z0-9]+' "$O/$tag.serve.out"); vram after load $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  python3 "$VC" vision-ab --url http://127.0.0.1:8097 --out "$O/$tag.check.json" > "$O/$tag.check.txt" 2>&1
  log "$tag vision_check rc=$?"; sed "s/^/$tag  /" "$O/$tag.check.txt" | tee -a "$O/run.log"
  # airbench answers stay out of the repo: raw replies in $RAW, the eval keeps reply_sha12 and reply_len
  python3 "$O/vision_timing.py" vision-ab --url http://127.0.0.1:8097 --images $AB/img --prompts $AB/s2_vision.json \
      --out "$RAW/$tag.timing.json" > "$RAW/$tag.timing.txt" 2>&1
  log "$tag timing rc=$?"; grep -E '^(summary|text probe)' "$RAW/$tag.timing.txt" | sed "s/^/$tag  /" | tee -a "$O/run.log"
  python3 - "$RAW/$tag.timing.json" "$O/$tag.timing.json" <<'PY'
import hashlib, json, sys
d = json.load(open(sys.argv[1]))
for r in d.get("results", []):
    if "reply" in r:
        rep = r.pop("reply"); r["reply_sha12"] = hashlib.sha256(rep.encode()).hexdigest()[:12]; r["reply_len"] = len(rep)
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
  python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke vision-ab --url http://127.0.0.1:8097 --long \
      --out "$O/$tag.smoke.json" > "$O/$tag.smoke.txt" 2>&1
  log "$tag smoke rc=$?"; grep -E '^(PASS|FAIL)' "$O/$tag.smoke.txt" | sed "s/^/$tag  /" | tee -a "$O/run.log"
  log "$tag vram peak $(awk -F= '{if($2>m)m=$2}END{print m}' "$O/$tag.mem.log") MiB; engine CPU fallbacks: $(grep -c 'encoded on the CPU' "$O/$tag.serve.out" $HOME/.cache/halocline/*/engine.log 2>/dev/null | awk -F: '{s+=$NF}END{print s+0}')"
  kill -TERM $SRV; wait $SRV 2>/dev/null; kill $TR 2>/dev/null
  sleep 5
done
log "done"
