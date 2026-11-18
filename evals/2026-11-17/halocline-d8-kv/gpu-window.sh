#!/usr/bin/env bash
# D8 GPU window on the d8-integration build: the GPU suite (gate for the release), then the D4 protocol
# (conv/struct/prose fixtures, prefill 16K/36K, 100-chunk KLD vs the 27B BF16 base) for int8, k8v4, k8v4 + --mtp-kv int8.
# Run under: berth train --as halocline-d8-gpu -- gpu-window.sh   (nothing else may hold the card; no nested berth run)
set -u
W=$HOME/wt/halocline-d8-int
OUT=$(cd "$(dirname "$0")" && pwd)
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
BASE=/data/models/halocline/kld-bases/qwen38-27b-bf16-base.kld
echo "$(date -Is) start, build $(git -C "$W" rev-parse --short HEAD)" | tee -a "$OUT/run.log"
nvidia-smi --query-gpu=power.limit,power.max_limit,driver_version --format=csv,noheader >> "$OUT/run.log"

# 1. GPU suite
( cd "$W/build" && HALO_QWEN35_PACK=$PACK ctest -L gpu -j 1 --output-on-failure ) > "$OUT/ctest-gpu.log" 2>&1
echo "$(date -Is) ctest-gpu rc=$? $(grep -E 'tests passed|tests failed' "$OUT/ctest-gpu.log" | tail -1)" | tee -a "$OUT/run.log"

# 2. D4 protocol arms (same flags as evals/2026-11-17/halocline-dense-quality/d4.sh, minus the berth wrapper)
arm() {
  local name=$1; shift
  mkdir -p "$OUT/$name"
  echo "$(date -Is) arm $name: $*" | tee -a "$OUT/run.log"
  "$W/bin/halocline-bench" --model "$PACK" -t conv,struct,prose -p 16384,36864 --seed 1 \
      --quality 100 --kld-base "$BASE" --logs "$OUT/$name" -o json -- --spec 8 --max-context 37528 "$@" \
      > "$OUT/$name/bench.json" 2> "$OUT/$name/bench.err"
  echo "$(date -Is) arm $name rc=$?" | tee -a "$OUT/run.log"
}
arm int8 --kv int8
arm k8v4 --kv k8v4
arm k8v4-mtpint8 --kv k8v4 --mtp-kv int8
echo "$(date -Is) end" | tee -a "$OUT/run.log"
