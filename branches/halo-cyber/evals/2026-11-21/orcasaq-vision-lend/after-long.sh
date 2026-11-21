#!/usr/bin/env bash
# The case that failed through llama-swap (2026-11-20, Orca: 15 pictures on the CPU in its engine log): pictures after a
# 137K prompt, then again with all but ~200 MiB of the card held by hog.py. Both dense packs on their served profiles,
# built from the lend branches (the encoder borrows pre_arena). Expect 3/3 known answers and 0 CPU-fallback lines.
# Run under: berth train --as halocline-vision-after-long -- ./after-long.sh [dense|orca ...]
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
log() { echo "$(date -Is) $*" | tee -a "$O/after-long.log"; }
exec 9>/tmp/halocline-build.lock
flock 9
for m in ${*:-dense orca}; do
  case $m in
    dense) W=$HOME/wt/halocline-vision-arena; P=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
           A=Halocline-27B-UD-Q4_K_XL; PROF="--spec 8 --long-context" ;;
    orca)  W=$HOME/wt/halocline-orca-lend; P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored-vision.halo
           A=Halocline-OrcaSAQ-27B; PROF="--spec 8 --max-context 262144 --kv k8v4 --mtp-kv int8 --mtp-window 32768" ;;
    *) log "unknown $m"; continue ;;
  esac
  log "$m start: $W HEAD $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --porcelain -uno | wc -l)"
  "$W/bin/halocline" serve "$P" --python "$PY" --host 127.0.0.1 --port 8097 --alias $A $PROF --vision dynamic \
      --draft-vocab "$W/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin" > "$O/al-$m.serve.out" 2>&1 &
  S=$!
  ( while kill -0 $S 2>/dev/null; do echo "$(date +%T) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 0.5; done ) > "$O/al-$m.mem.log" 2>&1 & TR=$!
  for t in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:8097/v1/models >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 1; done
  ELOG=$(ls -t $HOME/.cache/halo/*/engine.log | head -1); base=$(grep -c 'encoded on the CPU' "$ELOG")
  log "$m ready, engine log $ELOG (fallback lines before: $base); vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke $A --url http://127.0.0.1:8097 --long --out "$O/al-$m.smoke.json" > "$O/al-$m.smoke.txt" 2>&1
  log "$m smoke rc=$?"; grep -E '^(PASS|FAIL)' "$O/al-$m.smoke.txt" | sed "s/^/$m  /" | tee -a "$O/after-long.log"
  python3 "$VC" $A --url http://127.0.0.1:8097 --out "$O/al-$m.after-long.json" > "$O/al-$m.after-long.txt" 2>&1
  log "$m after the long prompt: rc=$?"; sed "s/^/$m  /" "$O/al-$m.after-long.txt" | tee -a "$O/after-long.log"
  KEEP_MIB=200 HOLD_S=120 python3 "$O/hog.py" > "$O/al-$m.hog.txt" 2>&1 & HOG=$!
  sleep 4; log "$m $(cat "$O/al-$m.hog.txt")"
  python3 "$VC" $A --url http://127.0.0.1:8097 --out "$O/al-$m.squeezed.json" > "$O/al-$m.squeezed.txt" 2>&1
  log "$m squeezed (~200 MiB free): rc=$?"; sed "s/^/$m  /" "$O/al-$m.squeezed.txt" | tee -a "$O/after-long.log"
  kill -TERM $HOG 2>/dev/null; wait $HOG 2>/dev/null
  log "$m CPU-fallback lines this run: $(( $(grep -c 'encoded on the CPU' "$ELOG") - base )); workspace lines: $(grep -c "in the prompt workspace" "$ELOG"); vram peak $(awk -F= '{if($2>m)m=$2}END{print m}' "$O/al-$m.mem.log") MiB"
  grep 'encoded on the CPU\|image encoder' "$ELOG" | tail -n +1 | tail -4 > "$O/al-$m.engine-lines.txt"
  kill -TERM $S; wait $S 2>/dev/null; kill $TR 2>/dev/null; sleep 5
done
log done
