#!/usr/bin/env bash
# Release candidate aa156951 (public main 5aec8e3a + W31 mapped vision projector + monitor redirect + W15 --draft-vocab
# + W33 planner scratch + docs). Run under: berth train --as halocline-rc-gate -- gate.sh [STEP ...]
# gpu    the GPU suite
# kld    dense --long-context depth KLD at 65,536 and 140,288 paired against the served kernel's arms
#        (../../2026-11-18/halocline-long-context lc-q4-*): W33 changes allocations only, so +0.00000; and the same at
#        65,536 with --draft-vocab 32K (verify scores every row): +0.00000
# ab     dense speed, the six workloads (decode.py), --long-context --vision cpu at 262,144: release 5aec8e3a (served)
#        against aa156951 with --draft-vocab 32K, order rel rc rc rel. The rc arm must LOAD at 262,144 (W33's point)
# vision the three known-answer images on the dense rc build (CPU encoder, now mapped)
# sparse the served sparse flags on the rc build from the NVMe pack with --vision cpu --ram-reserve-mib 2048 (W31's
#        point): smoke --long, the sidecar's smaps (file-backed vs anonymous), MemAvailable through the smoke, images
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
export TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
B=$HOME/halocline/releases/aa156951
R=$HOME/halocline/releases/5aec8e3a
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
PACKV=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
SPARSE=/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
V32=$B/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin
D=/data/models/halocline/kld-bases/depth
LC=$O/../../2026-11-18/halocline-long-context
PAIRED=$O/../../2026-11-17/halocline-kv-kld/kv_paired.py
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
wait_loaded() {  # $1 pid, $2 port
  local st=""
  for t in $(seq 1 450); do
    kill -0 $1 2>/dev/null || break
    st=$(curl -s -m 5 http://127.0.0.1:$2/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
    [ "$st" = loaded ] && return 0; sleep 2
  done
  return 1
}
log "start $(cat $B/RELEASE_COMMIT | cut -c1-80); power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
for s in ${*:-gpu kld ab vision sparse}; do
  case $s in
    gpu)
      ctest --test-dir "$B/build-89" -L gpu -j 1 --output-on-failure > "$O/ctest-gpu.log" 2>&1
      log "gpu suite rc=$? $(grep -E 'tests passed|tests failed' "$O/ctest-gpu.log" | tail -1)" ;;
    kld)
      for arm in "rc-q4-65536:65536:" "rc-q4-140288:140288:" "rc-q4-65536-v32k:65536:--draft-vocab $V32"; do
        n=${arm%%:*}; rest=${arm#*:}; depth=${rest%%:*}; extra=${rest#*:}
        mkdir -p "$O/$n"; [ -e "$O/lc-q4-$depth" ] || ln -s "$LC/lc-q4-$depth" "$O/lc-q4-$depth"
        "$B/bin/halocline-bench" --model "$PACK" --quality 16 --kld-base "$D/qwen38-27b-bf16-depth$depth-w1024-c16.kld" \
            --logs "$O/$n" -o json -- --spec 8 --max-context $((depth + 1033)) --long-context $extra \
            > "$O/$n/bench.json" 2> "$O/$n/bench.err"
        log "$n rc=$?"
        python3 "$PAIRED" "$O" "lc-q4-$depth" "$n" >> "$O/paired.txt" 2>&1
      done
      log "kld paired: $(grep -E 'rc-q4' "$O/paired.txt" | grep -oE 'rc-q4-[0-9a-z-]+ .*\+[0-9.]+ \[[^]]*\]' | tr '\n' ';')" ;;
    ab)
      for arm in rel rc rc rel; do
        i=$(( $(find "$O" -maxdepth 1 -name 'ab-*.json' | wc -l) + 1 ))
        if [ $arm = rel ]; then bin=$R/bin/halocline; extra=(); else bin=$B/bin/halocline; extra=(--draft-vocab "$V32"); fi
        "$bin" serve "$PACKV" --python "$PY" --host 127.0.0.1 --port 8096 --alias rc --spec 8 --long-context --vision cpu \
            "${extra[@]}" > "$O/ab-$i-$arm.serve.out" 2>&1 &
        SRV=$!
        if ! wait_loaded $SRV 8096; then
          log "ab $i $arm FAILED to load: $(grep -m1 -i error "$O/ab-$i-$arm.serve.out")"; kill -TERM $SRV 2>/dev/null; wait $SRV; continue
        fi
        python3 "$O/decode.py" "$O/ab-$i-$arm.json" > "$O/ab-$i-$arm.log" 2>&1
        log "ab $i $arm rc=$? ctx=$(grep -oE 'ctx=[0-9]+' "$O/ab-$i-$arm.serve.out" | head -1) $(python3 -c 'import json,sys; print(" ".join("%s=%s(%s/%s)" % (r["work"], r["decode_tps"], r.get("drafts"), r.get("offered")) for r in json.load(open(sys.argv[1]))))' "$O/ab-$i-$arm.json" 2>/dev/null)"
        kill -TERM $SRV; wait $SRV; sleep 3
      done ;;
    vision)
      "$B/bin/halocline" serve "$PACKV" --python "$PY" --host 127.0.0.1 --port 8096 --alias rc-dense --spec 8 --long-context \
          --vision cpu --draft-vocab "$V32" > "$O/vision-dense.serve.out" 2>&1 &
      SRV=$!
      if wait_loaded $SRV 8096; then
        python3 "$O/vision_check.py" rc-dense --url http://127.0.0.1:8096 --out "$O/vision-dense.json" > "$O/vision-dense.txt" 2>&1
        log "vision dense rc=$? $(grep -c PASS "$O/vision-dense.txt")/3 PASS"
      else log "vision dense FAILED to load"; fi
      kill -TERM $SRV; wait $SRV; sleep 3 ;;
    sparse)
      "$B/bin/halocline" serve "$SPARSE" --python "$PY" --host 127.0.0.1 --port 8096 --alias rc-sparse --spec 8 --prefill auto \
          --max-context 262144 --kv int8 --vision cpu --ram-reserve-mib 2048 > "$O/sparse.serve.out" 2>&1 &
      SRV=$!
      ( while kill -0 $SRV 2>/dev/null; do echo "$(date +%T) avail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; sleep 2; done ) > "$O/sparse.ram.log" &
      if wait_loaded $SRV 8096; then
        python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke rc-sparse --url http://127.0.0.1:8096 --long --out "$O/sparse-smoke.json" > "$O/sparse-smoke.txt" 2>&1
        log "sparse smoke rc=$? $(grep -E 'needle|long' "$O/sparse-smoke.txt" | tr '\n' ' ')"
        python3 "$O/vision_check.py" rc-sparse --url http://127.0.0.1:8096 --out "$O/vision-sparse.json" > "$O/vision-sparse.txt" 2>&1
        log "vision sparse rc=$? $(grep -c PASS "$O/vision-sparse.txt")/3 PASS"
        vp=$(pgrep -f "halocline-vision --model $SPARSE" | head -1)
        if [ -n "$vp" ]; then
          awk '/^[0-9a-f]+-[0-9a-f]+ /{name=$6} /^Rss:/{if (name ~ /\.halo$/) f+=$2; else a+=$2} END{printf "sidecar rss: pack mapping %d MiB, everything else %d MiB\n", f/1024, a/1024}' /proc/$vp/smaps > "$O/sparse-sidecar-smaps.txt"
          log "$(cat "$O/sparse-sidecar-smaps.txt")"
        fi
        log "sparse min MemAvailable $(sort -t= -k2 -n "$O/sparse.ram.log" | head -1)"
      else log "sparse FAILED to load: $(grep -m1 -i error "$O/sparse.serve.out")"; fi
      kill -TERM $SRV; wait $SRV ;;
  esac
done
log "end"
