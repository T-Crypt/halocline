#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin
# 06:20 morning review: window 6 lost every 131K arm to context_overflow (prompt 133,205 > --max-context 133,120).
# The 131K ABBA alone, head vs kv prompt order, int8 and K8V4, with --max-context 135168. Run under berth train.
set -u
O=$(cd "$(dirname "$0")" && pwd)
W=$HOME/wt/halocline-prompt-l2
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.halo
echo "$(date -Is) window6b start, build $(git -C "$W" rev-parse --short HEAD)" | tee -a "$O/run.log"
for kv in int8 k8v4; do
  for arm in head-a kv-a kv-b head-b; do
    mkdir -p "$O/131k-$kv-$arm"
    HALO_PROMPT_ORDER=${arm%-*} "$W/bin/halocline-bench" --model "$PACK" -p 131072 --seed 1 --logs "$O/131k-$kv-$arm" -o json -- \
        --spec 8 --max-context 135168 --kv $kv --mtp-kv int8 > "$O/131k-$kv-$arm/bench.json" 2> "$O/131k-$kv-$arm/bench.err"
    echo "$(date -Is) 131k-$kv-$arm rc=$? $(grep -o 'pp131072 [0-9]* tokens, [0-9]* t/s engine' "$O/131k-$kv-$arm/bench.err" | tail -1)" | tee -a "$O/run.log"
  done
done
echo "$(date -Is) window6b end" | tee -a "$O/run.log"
