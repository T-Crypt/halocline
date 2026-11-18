#!/usr/bin/env bash
# Window 2 (build 7e0f43c4: K8V4 sign fix + 8-warp prompt shapes):
#   0. build the binaries (the card is ours, the previous window's engines are gone)
#   1. the K8V4 / int8 / drafter GPU tests on the real pack; the parity test's kernel bench runs on a quiet card here
#   2. prefill, int8 vs K8V4, ABBA, D4's depths (16K, 36K), conv decode only as the smallest fixture
#   3. the K8V4 depth base (BF16, depth 174,080)
set -u
O=$(cd "$(dirname "$0")" && pwd)
W=$HOME/wt/halocline-d8-int
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
echo "$(date -Is) window2 start, build $(git -C "$W" rev-parse --short HEAD)" | tee -a "$O/run.log"
flock /tmp/halocline-build.lock systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- \
    nice -n 19 cmake --build "$W/build" -j6 > "$O/window2-build.log" 2>&1
echo "$(date -Is) build rc=$?" | tee -a "$O/run.log"

( cd "$W/build" && HALO_QWEN35_PACK=$PACK ctest -R "qwen35_mtp_cache_test|gqa_k8v4_parity|gqa_attn_parity" --output-on-failure ) \
    > "$O/ctest-gpu-fixed.log" 2>&1
echo "$(date -Is) fixed tests rc=$? $(grep -E 'tests passed' "$O/ctest-gpu-fixed.log")" | tee -a "$O/run.log"

pf() {
  local name=$1; shift
  mkdir -p "$O/$name"
  "$W/bin/halocline-bench" --model "$PACK" -t conv -p 16384,36864 --seed 1 --logs "$O/$name" -o json -- \
      --spec 8 --max-context 37528 "$@" > "$O/$name/bench.json" 2> "$O/$name/bench.err"
  echo "$(date -Is) prefill $name rc=$?" | tee -a "$O/run.log"
}
pf pf-int8-a --kv int8
pf pf-k8v4-a --kv k8v4
pf pf-k8v4-b --kv k8v4
pf pf-int8-b --kv int8

"$HOME/handoffs/halocline/2026-11-17-workers/run-depth-bases.sh" 174080:12
echo "$(date -Is) window2 end" | tee -a "$O/run.log"
