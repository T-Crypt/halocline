#!/usr/bin/env bash
# Halocline KV sweep: per KV format, decode fixtures + 32K/131K prefill + quality gate (100 chunks) vs the BF16 base.
# One Berth lease per KV so the card is free between runs. Usage: kv-sweep.sh PACK OUTDIR [kv ...]
set -u
PACK=$1; OUT=$2; shift 2
KVS=${*:-int8 fp16 k8v4 q4_0}
REL=$HOME/halocline/releases/d314a21a
BASE=/data/models/halocline/kld-bases/flashnext-bf16-base.kld
mkdir -p "$OUT"
for kv in $KVS; do
  if pgrep -f "[e]ngine strata" >/dev/null; then echo "STRATA UP, stopping before $kv" | tee -a "$OUT/sweep.log"; exit 3; fi
  d=$OUT/$kv; mkdir -p "$d"
  echo "$(date -Is) start $kv" | tee -a "$OUT/sweep.log"
  ~/.local/bin/berth run --as halocline-quality --priority foreground --pinned -- \
    "$REL/bin/basalt-bench" --model "$PACK" -t conv,struct,prose -p 32768,131072 --seed 1 \
      --quality 100 --kld-base "$BASE" --logs "$d" -o json -- --kv "$kv" --spec 8 --prefill auto \
    > "$d/bench.json" 2> "$d/bench.err"
  rc=$?
  echo "$(date -Is) end $kv rc=$rc" | tee -a "$OUT/sweep.log"
done
echo "$(date -Is) sweep done" | tee -a "$OUT/sweep.log"
