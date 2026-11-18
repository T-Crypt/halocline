#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin   # systemd user units lack ~/.local/bin (berth, notify-phone)
# Window 4 (overnight, card booked b44): every dense depth-KLD arm, then the 261,119 BF16 depth base (n_ctx 262,144),
# then Q4 at that depth. Run under: berth train --as halocline-d8-w4 -- window4.sh
set -u
O=$(cd "$(dirname "$0")" && pwd)
echo "$(date -Is) window4 start" | tee -a "$O/run.log"
# 0. sparse gate 2's 131K prefill again, ABBA, prefill only (19:03: in window 3 it stalled at ~133K tokens on BOTH builds
#    with 4 GiB RAM available and zram full; the two runs that finished read 1,841 / 2,428 against the published 6,019)
SG=$HOME/homelab/state/evals/2026-11-17/halocline-sparse-regress/run-release-b22ecbb2/2b-pp131k
mkdir -p "$SG"
# quiet host for the ABBA (operator 19:05: "free up RAM before the 131K rerun"): pause the periodic user jobs, restore
# them after step 0 (blog-forks.timer was stopped by hand at 19:06: its extract_forks.py had run 2h39m holding ~1.2 GB)
QUIET="blog-berth.timer blog-forks.timer blog-herald.timer blog-loads.timer blog-projects.timer claude-peer-sync.timer
       holodeck-poster-sweep.timer hq-hygiene.timer qwen4-watch.timer skills-gate-sync.timer aphotic-agent-usage.timer
       aphotic-greeter-sync.timer beketov-bounties.timer lab-sync.timer"
systemctl --user stop $QUIET
for u in $QUIET; do systemctl --user stop "${u%.timer}.service" 2>/dev/null; done
sync; sleep 5
{ date -Is; free -m; swapon --show; ps -eo pid,rss,comm --sort=-rss | head -12; } > "$SG/memory-before.txt"
avail=$(free -g | awk '$1 == "Mem:" { print $7 }')
if [ "${avail:-0}" -ge 50 ]; then
  for pair in ref-a new-a new-b ref-b; do
    case $pair in ref*) b=$HOME/halocline/releases/d314a21a/bin/basalt-bench ;; *) b=$HOME/halocline/releases/b22ecbb2/bin/halocline-bench ;; esac
    mkdir -p "$SG/$pair"
    "$b" --model /data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt -p 131072 --seed 1 --logs "$SG/$pair" -o json -- \
        --kv int8 --spec 8 --prefill auto > "$SG/$pair/bench.json" 2> "$SG/$pair/bench.err"
    echo "$(date -Is) sparse pp131k $pair rc=$? avail $(free -g | awk '$1 == "Mem:" { print $7 }') GiB" | tee -a "$O/run.log"
  done
else
  echo "$(date -Is) sparse pp131k skipped: ${avail} GiB available (< 50)" | tee -a "$O/run.log"
fi
systemctl --user start $QUIET; echo "$(date -Is) quiet-host timers restored" | tee -a "$O/run.log"
"$O/run.sh" int8-65536 k8v4-65536 q4-65536 q4-d4 int8-140288 k8v4-140288 k8v4-174080
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
echo "$(date -Is) window4 end" | tee -a "$O/run.log"
