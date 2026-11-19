#!/usr/bin/env bash
# W22 (branch q4-decode-kernel): the staged decode kernel for a rotated q4_0 cache. Run under:
#   berth train --as halocline-w22-gate -- gate.sh [STEP ...]   (default: gpu kld ab)
# gpu  the GPU suite (gqa_k8v4_parity: the engine's Q4 launch against the tiles, bit for bit, every width)
# kld  --long-context depth KLD at 65,536 and 140,288 paired against ../halocline-long-context's lc-q4 arms (the tiled
#      kernel): must be +0.00000
# ab   speed, client-side: the six workloads with thinking off (decode.py), --long-context, release ea5868de against
#      this branch, order rel w22 w22 rel, a fresh server per arm
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
B=$HOME/wt/halocline-long-ctx
R=$HOME/halocline/releases/ea5868de
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
PACKV=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
D=/data/models/halocline/kld-bases/depth
LC=$O/../halocline-long-context
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
log "start $(git -C "$B" rev-parse --short HEAD)"
for s in ${*:-gpu kld ab}; do
  case $s in
    gpu)
      ctest --test-dir "$B/build-89" -L gpu -j 1 --output-on-failure > "$O/ctest-gpu.log" 2>&1
      log "gpu suite rc=$? $(grep -E 'tests passed|tests failed' "$O/ctest-gpu.log" | tail -1)"
      "$B/build-89/gqa_k8v4_parity" > "$O/parity.txt" 2>&1; log "parity rc=$? $(grep -E 'Q4 decode, the engine' "$O/parity.txt")" ;;
    kld)
      for depth in 65536 140288; do
        n=w22-q4-$depth; mkdir -p "$O/$n"
        [ -e "$O/lc-q4-$depth" ] || ln -s "$LC/lc-q4-$depth" "$O/lc-q4-$depth"
        "$B/bin/halocline-bench" --model "$PACK" --quality 16 --kld-base "$D/qwen38-27b-bf16-depth$depth-w1024-c16.kld" \
            --logs "$O/$n" -o json -- --spec 8 --max-context $((depth + 1033)) --long-context \
            > "$O/$n/bench.json" 2> "$O/$n/bench.err"
        log "$n rc=$?"
        python3 "$O/../../2026-11-17/halocline-kv-kld/kv_paired.py" "$O" "lc-q4-$depth" "$n" >> "$O/paired.txt" 2>&1
      done ;;
    ab)
      for arm in rel w22 w22 rel; do
        i=$(( $(find "$O" -maxdepth 1 -name 'ab-*.json' | wc -l) + 1 ))
        bin=$R/bin/halocline; [ $arm = w22 ] && bin=$B/bin/halocline
        "$bin" serve "$PACKV" --python "$PY" --host 127.0.0.1 --port 8096 --alias w22 --spec 8 --long-context --vision cpu \
            > "$O/ab-$i-$arm.serve.out" 2>&1 &
        SRV=$!
        st=""
        for t in $(seq 1 300); do
          kill -0 $SRV 2>/dev/null || break   # the server exited (a load refused): no point waiting
          st=$(curl -s -m 5 http://127.0.0.1:8096/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
          [ "$st" = loaded ] && break; sleep 2
        done
        if [ "$st" != loaded ]; then
          log "ab $i $arm FAILED to load: $(grep -m1 -i error "$O/ab-$i-$arm.serve.out")"; kill -TERM $SRV 2>/dev/null; wait $SRV; continue
        fi
        python3 "$O/decode.py" "$O/ab-$i-$arm.json" > "$O/ab-$i-$arm.log" 2>&1
        log "ab $i $arm rc=$? $(python3 -c 'import json,sys; print(" ".join("%s=%s" % (r["work"], r["decode_tps"]) for r in json.load(open(sys.argv[1]))))' "$O/ab-$i-$arm.json" 2>/dev/null)"
        kill -TERM $SRV; wait $SRV; sleep 3
      done ;;
  esac
done
log "end"
