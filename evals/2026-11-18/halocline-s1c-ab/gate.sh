#!/usr/bin/env bash
# S1 W17 (Halocline branch s1c-decode-kernel 2e277290: a staged K8V4 decode kernel, 8-12 warps, every T <= 8):
# the full GPU suite, the bench at short and deep cells, then the greedy ABBA A/B against release b22ecbb2.
# Run under: berth train --as halocline-s1c-gate -- gate.sh
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
S=$HOME/wt/halocline-s1c
log() { echo "$(date -Is) $*" | tee -a "$O/gate.log"; }
log "start $(git -C "$S" rev-parse --short HEAD)"
ctest --test-dir "$S/build-w" -L gpu -j 1 --output-on-failure > "$O/ctest-gpu.log" 2>&1
log "ctest gpu rc=$? $(grep -E 'tests passed|tests failed' "$O/ctest-gpu.log" | tail -1)"
( for d in 300 1200 4096 137404; do "$S/build-w/gqa_decode_bench" $d 100; done ) > "$O/bench.txt" 2>&1
log "bench rc=$?"
python3 "$O/ab.py" "$O/ab" rel=$HOME/halocline/releases/b22ecbb2 s1c=$S s1c=$S rel=$HOME/halocline/releases/b22ecbb2 > "$O/ab.log" 2>&1
log "ab rc=$?"
log "end"
