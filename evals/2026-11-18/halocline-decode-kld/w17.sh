#!/usr/bin/env bash
# W17 (s1c-decode-kernel 2e277290): the K8V4 parity test's own lines (bitwise or not against the tiles), then the
# depth-KLD gate (kld.sh) at 65,536 and 140,288 against the 2026-11-17 K8V4 reference. Run under berth train.
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
"$HOME/wt/halocline-s1c/build-w/gqa_k8v4_parity" > "$O/w17-parity.txt" 2>&1
echo "parity rc=$?" >> "$O/w17-parity.txt"
"$O/kld.sh" s1c "$HOME/wt/halocline-s1c"
