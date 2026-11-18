#!/usr/bin/env bash
# Depth-KLD gate for a decode-kernel change (operator 2026-11-18: a non-bitwise decode kernel is allowed if the paired
# depth KLD at 65,536 and 140,288 is no worse than today's K8V4). Same protocol and pack as
# ../../2026-11-17/halocline-d8-depth (k8v4-65536, k8v4-140288 there are the reference rows).
# Usage: berth train --as halocline-decode-kld -- kld.sh LABEL BUILD_DIR
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
L=$1 B=$2
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
D=/data/models/halocline/kld-bases/depth
REF=$O/../../2026-11-17/halocline-d8-depth
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
log "start $L $(git -C "$B" rev-parse --short HEAD)"
for depth in 65536 140288; do
  n=$L-k8v4-$depth; mkdir -p "$O/$n"
  [ -e "$O/ref-k8v4-$depth" ] || ln -s "$REF/k8v4-$depth" "$O/ref-k8v4-$depth"
  "$B/bin/halocline-bench" --model "$PACK" --quality 16 --kld-base "$D/qwen38-27b-bf16-depth$depth-w1024-c16.kld" \
      --logs "$O/$n" -o json -- --spec 8 --max-context $((depth + 1033)) --kv k8v4 --mtp-kv int8 \
      > "$O/$n/bench.json" 2> "$O/$n/bench.err"
  log "$n rc=$? $(python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; print("kld %.5f top1 %.2f ppl %.4f" % (q["kld_mean"], q["top1_pct"], q["ppl"]))' "$O/$n/bench.json" 2>/dev/null)"
  python3 "$O/../../2026-11-17/halocline-kv-kld/kv_paired.py" "$O" "ref-k8v4-$depth" "$n" >> "$O/paired.txt" 2>&1
done
log "end"
