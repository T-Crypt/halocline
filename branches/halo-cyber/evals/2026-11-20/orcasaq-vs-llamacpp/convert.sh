#!/usr/bin/env bash
# OrcaSAQ-2 Cyber 27B Uncensored (OrcaRouter's GGUF, sha256 4ab6bdf8...) -> a Halocline dense pack, every tensor as
# stored, no vision (the GGUF has no projector). Branch halo-cyber; converter from that checkout (= main 6ba99fcf).
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
G=/data/models/llm/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored.halo
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
got=$(sha256sum "$G" | cut -d' ' -f1); [ "$got" = 4ab6bdf8a9008869abf1630fb92d04b5439805f96b56692962fa240b1d439977 ] || { log "STOP: sha256 $got"; exit 1; }
log "sha256 ok"
cd "$HOME/wt/halocline-halo-cyber"
CUDA_VISIBLE_DEVICES= systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
  $PY tools/convert/convert.py --gguf "$G" --no-vision --name OrcaSAQ-2-27B-Uncensored --out "$P" > "$O/convert.txt" 2>&1 \
  || { log "STOP: convert $(tail -2 "$O/convert.txt" | tr '\n' ' ')"; exit 1; }
log "pack $(stat -c %s "$P") bytes"
CUDA_VISIBLE_DEVICES= $PY tools/convert/gguf_qwen35.py verify "$P" "$G" > "$O/verify.txt" 2>&1; log "verify rc=$? $(tail -1 "$O/verify.txt")"
log "done"
