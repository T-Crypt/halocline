#!/usr/bin/env bash
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
# NInfer target-token NLL on the depth bases' rows (prep.py). Served engine and artifact (llama-swap NInfer-Qwen3.8-27B):
# ninfer-*-v3-be68237c, qwen3_8_27b.v3.ninfer. Arms: served KV rk4v4-e8 at 65,536 and 140,288; int8 KV at 65,536.
# Run under: berth train --as ninfer-depth-ppl -- run.sh
set -u
O=$(cd "$(dirname "$0")" && pwd)
BIN=/mnt/models/ninfer-v3-artifacts/bin/ninfer-perplexity-v3-be68237c
MODEL=/mnt/models/ninfer-v3-artifacts/qwen3_8_27b.v3.ninfer
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
arm() {  # name corpus-dir kv
  local name=$1 dir=$2 kv=$3 ctx
  ctx=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["context"])' "$O/$dir/prep.json")
  log "arm $name: --context $ctx --stride 1024 --kv-dtype $kv"
  "$BIN" "$MODEL" --corpus "$O/$dir/manifest.json" --context "$ctx" --stride 1024 --kv-dtype "$kv" --output "$O/$name" \
      > "$O/$name.out" 2> "$O/$name.err"
  log "arm $name rc=$? $(tail -2 "$O/$name.out" | tr '\n' ' ')"
}
log "start"
arm rk4v4e8-65536 d65536 rk4v4-e8
arm rk4v4e8-140288 d140288 rk4v4-e8
arm int8-65536 d65536 int8
log "end"
notify-phone --level done --title "NInfer depth PPL done" "aphotic: $(grep -c 'rc=0' "$O/run.log") of 3 arms rc=0; Claude tabulates against Halocline next"
