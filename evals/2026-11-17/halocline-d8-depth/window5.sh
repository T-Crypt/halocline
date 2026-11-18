#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin
# Window 5 (window 4 stopped after q4-d4): the depth arms window 4 lost to the context_overflow bug (max-context depth + 1,025; now depth + 1,033),
# then the 261,119 base and q4 at 261,119 (moved here from window 4 so the fixed arms prove the fix first).
set -u
O=$(cd "$(dirname "$0")" && pwd)
echo "$(date -Is) window5 start" | tee -a "$O/run.log"
"$O/run.sh" int8-65536 k8v4-65536 q4-65536 int8-140288 k8v4-140288 k8v4-174080
B=/data/models/halocline/kld-bases/depth/qwen38-27b-bf16-depth261119-w1024-c16.kld
R=$HOME/handoffs/halocline/2026-11-17-workers/run-depth-bases.sh
chk() { [[ -s $B ]] && python3 -I $HOME/wt/halocline-import/tools/depth_kld/depth_kld.py check "$B" > "$O/base261119-check.txt" 2>&1; }
for ngl in 12 8 5; do   # 174,080 fit at ngl 12 with 7.5 GiB free; the 262K KV needs ~1 GiB more on the card
  echo "$(date -Is) base 261119 ngl $ngl" | tee -a "$O/run.log"
  "$R" "261119:$ngl" >> "$O/base261119.out" 2>&1
  if chk; then echo "$(date -Is) base 261119 ok (ngl $ngl): $(tail -1 "$O/base261119-check.txt")" | tee -a "$O/run.log"; break; fi
  echo "$(date -Is) base 261119 failed at ngl $ngl" | tee -a "$O/run.log"; rm -f "$B"
done
chk && "$O/run.sh" q4-261119
echo "$(date -Is) window5 end" | tee -a "$O/run.log"
