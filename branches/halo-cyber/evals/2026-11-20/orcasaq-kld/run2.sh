#!/usr/bin/env bash
# Does Halocline's served 262K profile (K8V4) cost OrcaSAQ-2 27B quality against llama.cpp's served profile (q8_0 K and
# V), at the same depth, on the same GGUF bytes? Reference: llama.cpp on that GGUF with f16 K and V (the most exact run of
# these weights), depth bases from tools/depth_kld (llama-kl-first.patch): a D-token prefix, then 1,024 scored rows,
# 16 chunks, wikitext-103 train. Arms scored against the same base:
#   llama-q8   llama.cpp, -ctk q8_0 -ctv q8_0 (the lab's served `cyber` KV)
#   halo-k8v4  Halocline halo-cyber, --kv k8v4 --mtp-kv int8 (the served Halocline-OrcaSAQ-27B KV)
#   halo-int8  Halocline, --kv int8 (65,536 only: the higher-precision KV where it fits)
# Steps skip when their output exists. Run under: flock /tmp/halocline-build.lock berth train --as halocline-orcasaq-kld -- ./run.sh
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
G=/data/models/llm/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored.halo
TXT=/data/models/halocline/kld-bases/depth/wikitext103-train-16MB.txt
LP=$HOME/wt/llama.cpp-klfirst/build-cuda/bin/llama-perplexity
HB=$HOME/wt/halocline-halo-cyber/bin/halocline-bench
BD=/data/models/halocline/kld-bases/depth/orcasaq
mkdir -p "$BD"
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
ngl() { [ "$1" -le 65536 ] && echo 99 || echo 48; }   # 140,288: f16 K/V 8.6 GiB, so 16 of 65 layers stay in host RAM
for D in ${DEPTHS:-65536 140288}; do
  B=$BD/orcasaq-f16kv-depth$D-w1024-c16.kld; C=$((D + 1025))
  if [ ! -s "$B" ]; then
    log "base $D (llama.cpp f16 K/V, -ngl $(ngl $D))"
    LLAMA_KL_FIRST=$D "$LP" -m "$G" -f "$TXT" -c $C --chunks 16 -b 4096 -ub 4096 -fa on -ngl $(ngl $D) -t 16 \
        --kl-divergence-base "$B" > "$O/base-$D.log" 2>&1 || { log "STOP base $D: $(tail -2 "$O/base-$D.log" | tr '\n' ' ')"; rm -f "$B"; exit 1; }
    log "base $D done $(stat -c %s "$B") bytes"
  fi
  a=llama-q8-$D
  if [ ! -s "$O/$a.log" ] || ! grep -q 'Mean    KLD' "$O/$a.log"; then
    log "arm $a"
    UB=4096; [ "$D" -gt 65536 ] && UB=1024   # 140,288: -ub 4096 needs a 5.3 GB compute buffer that does not fit (the served cyber entry runs -ub 1024)
    LLAMA_KL_FIRST=$D "$LP" -m "$G" -f "$TXT" -c $C --chunks 16 -b 4096 -ub $UB -fa on -ngl 99 -t 16 -ctk q8_0 -ctv q8_0 \
        --kl-divergence-base "$B" --kl-divergence > "$O/$a.log" 2>&1
    log "arm $a rc=$? $(grep -E 'Mean +KLD|Same top p|Mean PPL\(Q\)' "$O/$a.log" | tr -s ' ' | tr '\n' ' ')"
  fi
  for kv in k8v4 int8; do
    [ "$kv" = int8 ] && [ "$D" -gt 65536 ] && continue
    a=halo-$kv-$D; [ -s "$O/$a/bench.json" ] && continue
    mkdir -p "$O/$a"; log "arm $a"
    extra=""; [ "$D" -gt 65536 ] && extra="--mtp-window 32768"
    "$HB" --model "$P" --quality 16 --kld-base "$B" --logs "$O/$a" -o json -- \
        --spec 8 --max-context $((D + 1033)) --kv $kv --mtp-kv int8 $extra > "$O/$a/bench.json" 2> "$O/$a/bench.err"
    log "arm $a rc=$? $(python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; print("kld %.5f top1 %.2f ppl %.4f base %.4f" % (q["kld_mean"], q["top1_pct"], q["ppl"], q["base_ppl"]))' "$O/$a/bench.json" 2>/dev/null)"
  done
done
log end
