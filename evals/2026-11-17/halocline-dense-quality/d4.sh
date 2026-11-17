#!/usr/bin/env bash
# D4/D5 dense: decode fixtures, prefill, quality gate (100 chunks) vs the 27B BF16 base. Usage: d4.sh PACK OUTDIR KV [extra engine args]
set -u
PACK=$1; OUT=$2; KV=$3; shift 3
W=$HOME/wt/halocline-import
BASE=/data/models/halocline/kld-bases/qwen38-27b-bf16-base.kld
mkdir -p "$OUT"
if pgrep -f "[e]ngine strata" >/dev/null; then echo "STRATA UP" | tee -a "$OUT/run.log"; exit 3; fi
echo "$(date -Is) start $PACK $KV $*" | tee -a "$OUT/run.log"
nvidia-smi --query-gpu=power.limit,power.max_limit,driver_version --format=csv,noheader >> "$OUT/run.log"
~/.local/bin/berth run --as halocline-dense-quality --priority foreground --pinned -- \
  "$W/bin/halo-bench" --model "$PACK" -t conv,struct,prose -p 16384,36864 --seed 1 \
    --quality 100 --kld-base "$BASE" --logs "$OUT" -o json -- --kv "$KV" --spec 8 "$@" \
  > "$OUT/bench.json" 2> "$OUT/bench.err"
echo "$(date -Is) end rc=$?" | tee -a "$OUT/run.log"
