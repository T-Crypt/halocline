#!/usr/bin/env bash
# The gates on branch vision-pre-arena after the thread default (2026-11-21, 5KWJVRF): no-GPU checks (tools/verify.sh
# with the build), the GPU suite with the 27B pack, and the served --vision cpu sidecar on every thread (one image).
# Run under: berth train --as halocline-vision-gates -- ./gates.sh
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
W=$HOME/wt/halocline-vision-arena
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
export TMPDIR=/var/tmp/halocline
log() { echo "$(date -Is) $*" | tee -a "$O/gates.log"; }
exec 9>/tmp/halocline-build.lock
flock 9
cd $W
log "HEAD $(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
PYTHON=$PY tools/verify.sh build-89 > "$O/gates-verify.txt" 2>&1; log "verify.sh rc=$?"; grep -E '^(PASS|FAIL)' "$O/gates-verify.txt" | tee -a "$O/gates.log"
( cd build-89 && HALO_QWEN35_PACK=$PACK ctest -L gpu -j 1 --output-on-failure > "$O/gates-ctest-gpu.txt" 2>&1 ); log "ctest gpu rc=$?"
grep -E 'tests passed|tests failed|\(Failed\)|\*\*\*' "$O/gates-ctest-gpu.txt" | tail -15 | tee -a "$O/gates.log"
log "cpu sidecar threads: serving --vision cpu from the branch"
bin/halocline serve "$PACK" --python "$PY" --host 127.0.0.1 --port 8097 --alias vision-ab --spec 8 --long-context --vision cpu \
    --draft-vocab $W/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin > "$O/gates-cpu.serve.out" 2>&1 &
SRV=$!
for t in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:8097/v1/models >/dev/null && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
log "sidecar: $(pgrep -af 'bin/halocline-vision' | grep -oE -- '--threads [0-9]+' || echo 'no --threads')"
python3 "$VC" vision-ab --url http://127.0.0.1:8097 --out "$O/gates-cpu.check.json" > "$O/gates-cpu.check.txt" 2>&1; log "vision_check rc=$?"
sed 's/^/cpu32  /' "$O/gates-cpu.check.txt" | tee -a "$O/gates.log"
kill -TERM $SRV; wait $SRV 2>/dev/null
log "done"
