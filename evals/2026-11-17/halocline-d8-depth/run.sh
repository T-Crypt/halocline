#!/usr/bin/env bash
# Dense depth KLD (D8): int8 vs K8V4 at depth, BF16 depth bases (1,024 scored rows x 16 chunks after a DEPTH-token
# prefix, tools/depth_kld). Quality only (no fixtures), the d8-integration build. Arms run in value order; each is
# self-contained, so a stop between arms keeps every finished arm. Drafter cache: --mtp-kv int8 everywhere (its cache
# never touches the verify logits: k8v4 vs k8v4-mtpint8 at 36K gave a bit-identical quality block); int8 at 140,288
# also needs the 32K drafter ring to fit (planner 144,641), which keeps verify identical as well (W3b).
# Q4 arms (operator, 2026-11-17 17:05: "run the Q4 real-pack KLD arm too if time allows"): the d8-q4 build
# (~/wt/halocline-d8-q4, merge 63eb5db7 + test-only aa17ca93), `--kv q4_0 --mtp-kv int8`. q4-d4 is the window-1 D4
# protocol at 36K (fixtures, prefill 16K/36K, 100 chunks vs the 27B BF16 base) beside int8/k8v4/k8v4-mtpint8 in
# ../halocline-d8-kv; q4-65536 pairs with int8/k8v4 at the same depth.
# --max-context: the request is the first depth + 1 tokens plus max_new 1,024, and the engine admits it only when
# prompt + max_new + 8 <= context (src/program/serve/request_image.cpp), so depth + 1,033. (20:07: depth + 1,025 failed
# every depth arm with context_overflow.) k8v4 at 174,080 needs 175,113 of its 175,629; q4 at 261,119 needs 262,152.
# Usage: berth train --as halocline-d8-depth -- run.sh [ARM ...]    (default: all, in order)
set -u
W=$HOME/wt/halocline-d8-int
WQ4=$HOME/wt/halocline-d8-q4
O=$(cd "$(dirname "$0")" && pwd)
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
D=/data/models/halocline/kld-bases/depth
base() { echo "$D/qwen38-27b-bf16-depth$1-w1024-c16.kld"; }
arm() {   # name depth engine-args...
  local name=$1 depth=$2; shift 2
  [[ -s $O/$name/rows.tsv ]] && { echo "$(date -Is) skip $name (rows exist)" | tee -a "$O/run.log"; return; }
  mkdir -p "$O/$name"
  echo "$(date -Is) arm $name depth $depth: $*" | tee -a "$O/run.log"
  "${BW:-$W}/bin/halocline-bench" --model "$PACK" --quality 16 --kld-base "$(base "$depth")" --logs "$O/$name" -o json -- \
      --spec 8 --max-context $((depth + 1033)) "$@" > "$O/$name/bench.json" 2> "$O/$name/bench.err"
  echo "$(date -Is) arm $name rc=$? $(python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; print("kld %.5f top1 %.2f ppl %.4f base %.4f" % (q["kld_mean"], q["top1_pct"], q["ppl"], q["base_ppl"]))' "$O/$name/bench.json" 2>/dev/null)" | tee -a "$O/run.log"
}
echo "$(date -Is) start, build $(git -C "$W" rev-parse --short HEAD), q4 build $(git -C "$WQ4" rev-parse --short HEAD)" | tee -a "$O/run.log"
nvidia-smi --query-gpu=power.limit,driver_version --format=csv,noheader >> "$O/run.log"
ARMS=${*:-"int8-65536 k8v4-65536 q4-65536 q4-d4 int8-140288 k8v4-140288 k8v4-174080"}
for a in $ARMS; do
  case $a in
    int8-65536)  arm $a 65536  --kv int8 --mtp-kv int8 ;;
    k8v4-65536)  arm $a 65536  --kv k8v4 --mtp-kv int8 ;;
    int8-140288) arm $a 140288 --kv int8 --mtp-kv int8 --mtp-window 32768 ;;
    k8v4-140288) arm $a 140288 --kv k8v4 --mtp-kv int8 ;;
    k8v4-174080) arm $a 174080 --kv k8v4 --mtp-kv int8 ;;
    q4-65536)    BW=$WQ4 arm $a 65536 --kv q4_0 --mtp-kv int8 ;;
    # 262,144 cells with Q4 need the drafter's 32K ring AND --mtp-kv int8 (planner: 53 MiB to spare); verify unchanged
    q4-261119)   BW=$WQ4 arm $a 261119 --kv q4_0 --mtp-kv int8 --mtp-window 32768 ;;
    q4-d4)
      if [[ -s $O/$a/rows.tsv ]]; then echo "$(date -Is) skip $a (rows exist)" | tee -a "$O/run.log"; continue; fi
      mkdir -p "$O/$a"
      echo "$(date -Is) arm $a: D4 36K --kv q4_0 --mtp-kv int8 (q4 build)" | tee -a "$O/run.log"
      "$WQ4/bin/halocline-bench" --model "$PACK" -t conv,struct,prose -p 16384,36864 --seed 1 \
          --quality 100 --kld-base /data/models/halocline/kld-bases/qwen38-27b-bf16-base.kld --logs "$O/$a" -o json -- \
          --spec 8 --max-context 37528 --kv q4_0 --mtp-kv int8 > "$O/$a/bench.json" 2> "$O/$a/bench.err"
      echo "$(date -Is) arm $a rc=$?" | tee -a "$O/run.log" ;;
    *) echo "unknown arm $a" | tee -a "$O/run.log" ;;
  esac
done
echo "$(date -Is) end" | tee -a "$O/run.log"
