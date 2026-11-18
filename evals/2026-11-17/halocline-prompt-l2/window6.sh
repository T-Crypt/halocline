#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin
# Window 6: prompt-l2 (KV-head-major prompt block order, after llama.cpp #32407). 1. bit-identity test; 2. prefill ABBA
# head vs kv order, int8 and K8V4, 16K-131K, dense UD-Q4_K_XL. Run under: berth train --as halocline-prompt-l2 -- window6.sh
set -u
O=$(cd "$(dirname "$0")" && pwd)
W=$HOME/wt/halocline-prompt-l2
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.halo
echo "$(date -Is) window6 start, build $(git -C "$W" rev-parse --short HEAD)" | tee -a "$O/run.log"
# 21:40 (operator: "fix the ring first when the card frees up"): the drafter ring fix and its reruns go first
$HOME/homelab/state/evals/2026-11-17/halocline-d8-depth/ringfix.sh; echo "$(date -Is) ringfix rc=$?" | tee -a "$O/run.log"
"$W/build/gqa_prompt_order_test" > "$O/order-test.log" 2>&1
rc=$?; echo "$(date -Is) gqa_prompt_order_test rc=$rc $(tail -1 "$O/order-test.log")" | tee -a "$O/run.log"
[ $rc -eq 0 ] || { echo "$(date -Is) window6 end (order test failed, no A/B)" | tee -a "$O/run.log"; exit 1; }
for kv in int8 k8v4; do
  for arm in head-a kv-a kv-b head-b; do
    mkdir -p "$O/$kv-$arm"
    HALO_PROMPT_ORDER=${arm%-*} "$W/bin/halocline-bench" --model "$PACK" -p 16384,36864,65536,131072 --seed 1 --logs "$O/$kv-$arm" -o json -- \
        --spec 8 --max-context 133120 --kv $kv --mtp-kv int8 > "$O/$kv-$arm/bench.json" 2> "$O/$kv-$arm/bench.err"
    echo "$(date -Is) $kv-$arm rc=$? $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join("%s %.0f" % (s["test"], s["tok_s_mean"]) for s in d["summary"]))' "$O/$kv-$arm/bench.json" 2>/dev/null)" | tee -a "$O/run.log"
  done
done
echo "$(date -Is) window6 end" | tee -a "$O/run.log"
notify-phone --level done --title "Halocline prompt-l2 A/B done" "aphotic: $(grep -E 'order_test|rc=' "$O/run.log" | sed -E 's/^[0-9T:-]+[-+][0-9:]+ //' | tr '\n' ';' | cut -c1-900)"
