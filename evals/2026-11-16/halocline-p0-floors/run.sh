#!/usr/bin/env zsh
# Halocline P0 speed floors (ticket the board), 2026-11-16. Same harnesses as the 2026-11-04 Strata A/B so the numbers
# compare: Strata 0.2.10 + DFlash2 on UD-IQ4_XS v3.1 (lsw smoke --long + standard-v1, 3 runs), then NInfer-Qwen3.8-27B
# through llama-swap (lsw smoke --long, 3 runs). One engine on the card at a time; nothing else heavy running.
set -u
out=${0:A:h}
cd ~/homelab
echo "START $(date -Is)"
zsh scripts/strata/ab_drafting.sh "$out/strata" ~/strata/releases/0.2.10 3 df2=$HOME/strata/strata-ud-iq4_xs.json 2>&1 | tee "$out/strata.log"
for n in 1 2 3; do
  echo "== ninfer $n $(date -Is)"
  python3 scripts/llama-swap/lsw.py smoke NInfer-Qwen3.8-27B --long --out "$out/ninfer-$n.json" > "$out/ninfer-$n.txt" 2>&1
  echo "ninfer smoke exit $? $n"
done
curl -s -X POST 127.0.0.1:9090/api/models/unload/NInfer-Qwen3.8-27B >/dev/null
echo "FLOORS_DONE $(date -Is)"
