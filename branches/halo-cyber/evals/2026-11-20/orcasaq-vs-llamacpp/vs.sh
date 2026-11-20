#!/usr/bin/env bash
# OrcaSAQ-2 Cyber 27B Uncensored, the same GGUF bytes in both engines, one RTX 4090 at 350 W, one engine at a time:
#   halo   Halocline halo-cyber build, byte-identical pack, 262,144 tokens, K8V4 KV (K int8, V rotated 4-bit), int8 drafter
#          cache, --spec 8 with the 32K draft head (verify scores the full vocabulary)
#   llama  llama.cpp, the lab's `cyber` entry verbatim: 188,416 tokens, q8_0 K and V, its MTP draft (n-max 4)
# decode.py: six workloads + needles (halo 180K and 250K; llama 180K, its ceiling is 188,416).
# Run under: flock /tmp/halocline-build.lock berth train --as halocline-orca-vs -- ./vs.sh
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
H=$HOME/wt/halocline-halo-cyber; RV=$HOME/halocline/releases/b2f81178
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored.halo
G=/data/models/llm/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf
log() { echo "$(date -Is) $*" | tee -a "$O/vs.log"; }
wait_up() { for t in $(seq 1 300); do kill -0 $1 2>/dev/null || return 1; curl -sf -m 3 http://127.0.0.1:8096/health >/dev/null && return 0
  st=$(curl -s -m 3 http://127.0.0.1:8096/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
  [ "$st" = loaded ] && return 0; sleep 2; done; return 1; }
vram() { while kill -0 $1 2>/dev/null; do nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu,clocks.sm --format=csv,noheader >> "$O/$2.gpu.csv"; sleep 2; done; }
log "start $(nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader)"
for arm in ${ARMS:-halo llama}; do
  case $arm in
    halo) "$H/bin/halocline" serve "$P" --python "$HOME/halocline/releases/b22ecbb2/.venv/bin/python" --host 127.0.0.1 --port 8096 \
            --alias Halocline-OrcaSAQ-27B --spec 8 --max-context 262144 --kv k8v4 --mtp-kv int8 --mtp-window 32768 \
            --draft-vocab "$RV/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin" > "$O/$arm.serve.out" 2>&1 &
          N="180000 250000" ;;
    llama) /usr/local/bin/llama-server -m "$G" -ngl 99 --jinja --chat-template-file /mnt/models/llama-swap/templates/qwen3.8-midsystem.jinja \
            -ub 1024 --ctx-size 188416 --spec-type draft-mtp --spec-draft-n-max 4 --reasoning on --reasoning-format deepseek \
            --reasoning-budget 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 -fa on -ctk q8_0 -ctv q8_0 \
            --parallel 1 --host 127.0.0.1 --port 8096 > "$O/$arm.serve.out" 2>&1 &
          N="180000" ;;
  esac
  S=$!
  wait_up $S || { log "$arm FAILED to start: $(grep -m2 -iE 'error|fail' "$O/$arm.serve.out" | tr '\n' ' ')"; kill $S 2>/dev/null; wait $S; continue; }
  vram $S $arm & VP=$!
  log "$arm up"
  python3 "$O/decode.py" http://127.0.0.1:8096 "$O/$arm.json" $N > "$O/$arm.log" 2>&1
  log "$arm rc=$? peak $(sort -t, -k1 -n "$O/$arm.gpu.csv" | tail -1)"
  kill -TERM $S; wait $S 2>/dev/null; kill $VP 2>/dev/null; sleep 5
done
log end
