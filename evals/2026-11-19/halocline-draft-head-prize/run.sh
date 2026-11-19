#!/usr/bin/env bash
# How much does W15's frequency-ranked draft head buy on the dense 262K profile? (the NInfer prose bar, standing order 6)
# It does not load at 262,144 yet (no VRAM, ../../2026-11-18/halocline-draft-head-lc), so every arm here runs at
# --max-context 245760 with the same --long-context cache: the six workloads reach 137K, so the cap does not touch them.
# Build: ~/wt/halocline-w15 (draft-head-lc 30104ad4, pre-W22 kernel: absolute rates sit below release 5aec8e3a's; read
# the arms against each other). Lossless: paired depth KLD +0.00000 (../../2026-11-18/halocline-draft-head-lc/paired.txt).
# Run under: berth train --as halocline-draft-head-prize -- run.sh   (order off v32k v64k v64k v32k off)
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
B=$HOME/wt/halocline-w15
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
V=$B/tools/draft_vocab/data
PACKV=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
export TMPDIR=/var/tmp/halocline
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
log "start $(git -C "$B" rev-parse --short HEAD); power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
i=0
for arm in ${*:-off v32k v64k v64k v32k off}; do
  i=$((i + 1)); extra=()
  [ $arm = v32k ] && extra=(--draft-vocab "$V/qwen38-27b-wikitext103-src-n32768.bin")
  [ $arm = v64k ] && extra=(--draft-vocab "$V/qwen38-27b-wikitext103-src-n65536.bin")
  "$B/bin/halocline" serve "$PACKV" --python "$PY" --host 127.0.0.1 --port 8096 --alias w15 --spec 8 --long-context \
      --max-context 245760 --vision cpu "${extra[@]}" > "$O/ab-$i-$arm.serve.out" 2>&1 &
  SRV=$!
  st=
  for t in $(seq 1 300); do
    st=$(curl -s -m 5 http://127.0.0.1:8096/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
    [ "$st" = loaded ] && break; kill -0 $SRV 2>/dev/null || break; sleep 2
  done
  if [ "$st" != loaded ]; then log "ab $i $arm did not load: $(grep -i error "$O/ab-$i-$arm.serve.out" | tail -1)"; kill -TERM $SRV 2>/dev/null; wait $SRV; continue; fi
  python3 "$O/decode.py" "$O/ab-$i-$arm.json" > "$O/ab-$i-$arm.log" 2>&1
  log "ab $i $arm rc=$? $(python3 -c 'import json,sys; print(" ".join("%s=%s(%s/%s)" % (r["work"], r["decode_tps"], r.get("drafts"), r.get("offered")) for r in json.load(open(sys.argv[1]))))' "$O/ab-$i-$arm.json" 2>/dev/null)"
  kill -TERM $SRV; wait $SRV; sleep 3
done
log end
