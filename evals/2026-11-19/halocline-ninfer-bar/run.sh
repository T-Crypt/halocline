#!/usr/bin/env bash
# The NInfer switch bar (Halocline roadmap standing order 6) on the served 262K profile, same session for both engines,
# plus the tool-call breadth gate (tools/tool_calls, 24 cases) on both. Through llama-swap :9090; llama-swap and Berth
# swap the engines. Release: whatever the Halocline-27B-UD-Q4_K_XL entry serves (recorded below).
#   speed  vs_ninfer_nothink.py: six workloads, thinking off, greedy, 1,200 tokens, streamed, order H N N H
#   tools  tool_calls.py run (thinking on as served, temperature 0, max_tokens 4096) on Halocline, then NInfer, then score each (Halocline scored against NInfer as --ref too)
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
T=$HOME/wt/halocline-readme/tools/tool_calls
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
# the card must be free: the run before (a berth train lease) releases a second after its last line (failed-race/)
for t in $(seq 1 120); do berth status 2>/dev/null | grep -q "no leases" && break; sleep 5; done
log "start; served: $(grep -A8 '"Halocline-27B-UD-Q4_K_XL":' $HOME/homelab/reference/llama-swap/config.yaml | grep -oE 'releases/[0-9a-f]+' | head -1); power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
for s in ${*:-speed tools}; do
  case $s in
    speed)
      python3 "$O/vs_ninfer_nothink.py" "$O/speed.json" > "$O/speed.log" 2>&1; log "speed rc=$?" ;;
    tools)
      for m in Halocline-27B-UD-Q4_K_XL NInfer-Qwen3.8-27B; do
        python3 "$T/tool_calls.py" run --url http://127.0.0.1:9090 --model $m --cases "$T/cases.jsonl" --max-tokens 4096 --out "$O/tools-$m.jsonl" > "$O/tools-$m.log" 2>&1
        log "tools $m rc=$?"
      done
      python3 "$T/tool_calls.py" score "$O/tools-NInfer-Qwen3.8-27B.jsonl" > "$O/tools-score-ninfer.txt" 2>&1
      python3 "$T/tool_calls.py" score "$O/tools-Halocline-27B-UD-Q4_K_XL.jsonl" --ref "$O/tools-NInfer-Qwen3.8-27B.jsonl" > "$O/tools-score-halocline.txt" 2>&1
      log "tools scored: $(tail -1 "$O/tools-score-halocline.txt")" ;;
  esac
done
log done
