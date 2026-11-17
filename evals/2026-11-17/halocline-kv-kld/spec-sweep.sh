#!/usr/bin/env bash
# Halocline drafter window sweep: decode fixtures only, one KV, per --spec T. Drafting is lossless (output = plain
# decoding), so T is chosen on speed alone. Usage: spec-sweep.sh PACK OUTDIR KV T [T ...]
set -u
PACK=$1; OUT=$2; KV=$3; shift 3
REL=$HOME/halocline/releases/d314a21a
mkdir -p "$OUT"
for t in "$@"; do
  if pgrep -f "[e]ngine strata" >/dev/null; then echo "STRATA UP, stopping before T=$t" | tee -a "$OUT/sweep.log"; exit 3; fi
  d=$OUT/spec$t; mkdir -p "$d"
  echo "$(date -Is) start spec $t" | tee -a "$OUT/sweep.log"
  ~/.local/bin/berth run --as halocline-quality --priority foreground --pinned -- \
    "$REL/bin/basalt-bench" --model "$PACK" -t conv,struct,prose -r 2 --seed 1 --logs "$d" -o json -- \
      --kv "$KV" --spec "$t" --prefill auto \
    > "$d/bench.json" 2> "$d/bench.err"
  echo "$(date -Is) end spec $t rc=$?" | tee -a "$OUT/sweep.log"
done
echo "$(date -Is) sweep done" | tee -a "$OUT/sweep.log"
