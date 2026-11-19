#!/usr/bin/env bash
# The 262K profile's gate (branch long-context: d8-q4 merged onto main with S1, --long-context). Run under:
#   berth train --as halocline-long-context -- gate.sh [STEP ...]   (default: all, in order; a finished arm is skipped)
# 1 gpu    the GPU suite (gqa_k8v4_parity carries the Q4 arm's fp64 references)
# 2 k8v4   the default cache stays bitwise: depth KLD at 65,536 and 140,288 paired against the 2026-11-17 K8V4 reference
# 3 q4     the 262K profile's quality, through --long-context itself: depth KLD at 65,536 / 140,288 / 174,080, each
#          paired against K8V4 at the same depth (the 65,536 arm also against d8-q4's own q4-65536: the port check)
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
B=$HOME/wt/halocline-long-ctx
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
D=/data/models/halocline/kld-bases/depth
REF=$O/../../2026-11-17/halocline-d8-depth
PAIR=$O/../../2026-11-17/halocline-kv-kld/kv_paired.py
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
quality() { python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; print("kld %.5f top1 %.2f ppl %.4f" % (q["kld_mean"], q["top1_pct"], q["ppl"]))' "$1" 2>/dev/null; }
arm() {   # name depth engine-args...
  local n=$1 depth=$2; shift 2
  [[ -s $O/$n/rows.tsv ]] && { log "skip $n (rows exist)"; return; }
  mkdir -p "$O/$n"
  log "arm $n depth $depth: $*"
  "$B/bin/halocline-bench" --model "$PACK" --quality 16 --kld-base "$D/qwen38-27b-bf16-depth$depth-w1024-c16.kld" \
      --logs "$O/$n" -o json -- --spec 8 --max-context $((depth + 1033)) "$@" > "$O/$n/bench.json" 2> "$O/$n/bench.err"
  log "arm $n rc=$? $(quality "$O/$n/bench.json")"
}
log "start $(git -C "$B" rev-parse --short HEAD); $(nvidia-smi --query-gpu=power.limit,driver_version --format=csv,noheader)"
for s in ${*:-gpu k8v4 q4}; do
  case $s in
    gpu)
      ctest --test-dir "$B/build-89" -L gpu -j 1 --output-on-failure > "$O/ctest-gpu.log" 2>&1
      log "gpu suite rc=$? $(grep -E 'tests passed|tests failed' "$O/ctest-gpu.log" | tail -1)" ;;
    k8v4)
      for depth in 65536 140288; do
        [ -e "$O/ref-k8v4-$depth" ] || ln -s "$REF/k8v4-$depth" "$O/ref-k8v4-$depth"
        arm lc-k8v4-$depth $depth --kv k8v4 --mtp-kv int8
        python3 "$PAIR" "$O" "ref-k8v4-$depth" "lc-k8v4-$depth" >> "$O/paired.txt" 2>&1
      done ;;
    q4)
      [ -e "$O/d8q4-q4-65536" ] || ln -s "$REF/q4-65536" "$O/d8q4-q4-65536"
      [ -e "$O/ref-k8v4-174080" ] || ln -s "$REF/k8v4-174080" "$O/ref-k8v4-174080"
      for depth in 65536 140288 174080; do
        [ -e "$O/ref-k8v4-$depth" ] || ln -s "$REF/k8v4-$depth" "$O/ref-k8v4-$depth"
        arm lc-q4-$depth $depth --long-context
        python3 "$PAIR" "$O" "ref-k8v4-$depth" "lc-q4-$depth" >> "$O/paired.txt" 2>&1
      done
      python3 "$PAIR" "$O" "d8q4-q4-65536" "lc-q4-65536" >> "$O/paired.txt" 2>&1 ;;
  esac
done
log "end"
