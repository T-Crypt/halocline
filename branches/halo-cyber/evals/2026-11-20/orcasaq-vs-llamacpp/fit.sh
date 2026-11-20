#!/usr/bin/env bash
# Which KV format holds OrcaSAQ-2 27B (byte-identical pack) at its native 262,144 tokens on the 4090? Highest precision
# first (operator: "go for higher KV"). Each arm: start the server, wait for loaded or exit, record VRAM, stop.
# Run under: flock /tmp/halocline-build.lock berth train --as halocline-orca-fit -- ./fit.sh
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
R=$HOME/wt/halocline-halo-cyber   # halo-cyber build: IQ4_XS gating tensors (adapter.cpp)
RV=$HOME/halocline/releases/b2f81178
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored.halo
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
V32=$RV/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin
log() { echo "$(date -Is) $*" | tee -a "$O/fit.log"; }
for arm in ${ARMS:-int8 k8v4}; do
  "$R/bin/halocline" serve "$P" --python "$PY" --host 127.0.0.1 --port 8097 --alias fit --spec 8 \
      --max-context 262144 --kv "$arm" --mtp-kv int8 --mtp-window 32768 --draft-vocab "$V32" > "$O/fit-$arm.out" 2>&1 &
  S=$!; st=
  for t in $(seq 1 240); do
    kill -0 $S 2>/dev/null || { st=exited; break; }
    v=$(curl -s -m 3 http://127.0.0.1:8097/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
    [ "$v" = loaded ] && { st=loaded; break; }; sleep 2
  done
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')
  log "$arm: ${st:-timeout} vram [$mem] $(grep -m1 -iE 'ctx=|context' "$O/fit-$arm.out" | cut -c1-160) | $(grep -iE 'error|refus|does not fit|out of memory' "$O/fit-$arm.out" | head -2 | tr '\n' ' ' | cut -c1-240)"
  kill -TERM $S 2>/dev/null; wait $S 2>/dev/null; sleep 3
done
log end
