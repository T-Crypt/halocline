#!/usr/bin/env bash
# OrcaSAQ-2 27B + the stock Qwen3.8-27B BF16 projector (Unsloth mmproj-BF16.gguf, sha256 83ee4f4f..., the same bytes as the
# served UD dense pack's vision section). Do images work on the abliterated weights? Served profile (262,144, K8V4,
# --spec 8), the three known-answer images (vision_check.py), encoder on CPU, then --vision dynamic (expected to refuse
# beside 262K, as the UD dense pack does). Halocline release 25d5cb79 (halo-cyber).
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
R=$HOME/halocline/releases/25d5cb79
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
for v in ${MODES:-cpu dynamic}; do
  "$R/bin/halocline" serve "$P" --python "$HOME/halocline/releases/b22ecbb2/.venv/bin/python" --host 127.0.0.1 --port 8096 \
      --alias Halocline-OrcaSAQ-27B --spec 8 --max-context 262144 --kv k8v4 --mtp-kv int8 --mtp-window 32768 --vision $v \
      --draft-vocab "$R/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin" > "$O/serve-$v.out" 2>&1 &
  S=$!; st=
  for t in $(seq 1 240); do kill -0 $S 2>/dev/null || { st=exited; break; }
    curl -s -m 3 http://127.0.0.1:8096/v1/models | grep -q '"loaded"' && { st=loaded; break; }; sleep 2; done
  if [ "$st" != loaded ]; then log "$v: ${st:-timeout} $(grep -m2 -iE 'error|refus' "$O/serve-$v.out" | tr '\n' ' ' | cut -c1-240)"; kill $S 2>/dev/null; wait $S 2>/dev/null; continue; fi
  python3 "$VC" Halocline-OrcaSAQ-27B --url http://127.0.0.1:8096 --out "$O/vision-$v.json" > "$O/vision-$v.txt" 2>&1
  log "$v: rc=$? $(grep -c '^PASS' "$O/vision-$v.txt")/3 PASS, vram $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
  kill -TERM $S; wait $S 2>/dev/null; sleep 4
done
log end
