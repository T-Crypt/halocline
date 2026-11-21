#!/usr/bin/env bash
# Round 5: both GPU failure modes (no arena; arena but cuBLAS workspace, status 13) fall back to the CPU. The sequence
# that failed through llama-swap: lsw smoke --long, then the three known-answer images, three times. halo-cyber build.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd); H=$HOME/wt/halocline-halo-cyber
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
log() { echo "$(date -Is) $*" | tee -a "$O/run5.log.txt"; }
"$H/bin/halocline" serve "$P" --python "$HOME/halocline/releases/b22ecbb2/.venv/bin/python" --host 127.0.0.1 --port 8096 \
    --alias Halocline-OrcaSAQ-27B --spec 8 --max-context 262144 --kv k8v4 --mtp-kv int8 --mtp-window 32768 --vision dynamic \
    --draft-vocab "$H/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin" > "$O/serve5.out" 2>&1 &
S=$!
for t in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:8096/v1/models | grep -q '"loaded"' && break; kill -0 $S || { log "server exited"; exit 1; }; sleep 2; done
for r in 1 2 3; do
  python3 "$HOME/homelab/scripts/llama-swap/lsw.py" smoke Halocline-OrcaSAQ-27B --url http://127.0.0.1:8096 --long > "$O/smoke5-$r.txt" 2>&1
  python3 "$VC" Halocline-OrcaSAQ-27B --url http://127.0.0.1:8096 --out "$O/vision5-$r.json" > "$O/vision5-$r.txt" 2>&1
  log "round $r: smoke $(grep -c '^PASS' "$O/smoke5-$r.txt")/5, images $(grep -c '^PASS' "$O/vision5-$r.txt")/3 ($(awk '{print $3}' "$O/vision5-$r.txt" | tr '\n' ' ')s), vram $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
done
kill -TERM $S; wait $S 2>/dev/null
L=$(find ~/.cache/halo -name engine.log -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
grep 'encoded on the CPU' "$L" | tail -9 > "$O/engine-fallback-lines5.txt"
log "fallback lines: $(wc -l < "$O/engine-fallback-lines5.txt") ($(grep -c 'no device memory' "$O/engine-fallback-lines5.txt") no arena, $(grep -c 'encoder failed' "$O/engine-fallback-lines5.txt") encoder failed)"
log end
