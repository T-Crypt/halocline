#!/usr/bin/env bash
# SC117 GSQ-RCO abliterated IQ3_S on Halocline (branch abil-halo), 2026-11-20. Operator: "Download IQ3_S SC117 as
# abil-halo", "bump the limit to 450 when IQ3_S lands".
# 1. sha256 of both shards against the HF LFS oids. 2. convert.py --dry-run, then the pack on /data (drafter and vision
# from the served UD pack). 3. halocline-bench --community: A UD@350, B SC117@350, C SC117@450, D UD@450, rows into the
# branch's benchmarks.json. 450 W only for C and D, under a thermal watchdog (the 350 W cap exists because the card hit
# 84-85 C on 2026-11-02); gpu-tune.service restores 350 W at the end.
# Run under nothing: it takes its own Berth lease for the GPU work. Log: run.log.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
SRC=/data/models/llm/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/IQ3_S
S1=$SRC/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf
S2=$SRC/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf
UD=/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
PACK=/data/models/halocline/packs/QFN-SC117-GSQ-RCO-abliterated-IQ3_S-vision.halo
R=$HOME/wt/halocline-abil-halo   # a git checkout: --community records the commit (release trees are git archives)
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
WT=$HOME/wt/halocline-abil-halo
TUNE=$HOME/homelab/scripts/aphotic/gpu_tune.py
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
die() { log "STOP: $*"; exit 1; }

# 1. checksums
want1=40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c
want2=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
if [ -f "$O/sha256.ok" ]; then log "sha256 already ok (sha256.ok)"; else
log "sha256 start"
got1=$(sha256sum "$S1" | cut -d' ' -f1); got2=$(sha256sum "$S2" | cut -d' ' -f1)
[ "$got1" = "$want1" ] || die "shard 1 sha256 $got1 != $want1"
[ "$got2" = "$want2" ] || die "shard 2 sha256 $got2 != $want2"
log "sha256 ok: both shards match the HF LFS oids"
touch "$O/sha256.ok"; fi

# 2. pack (CPU and disk only; the served sparse engine is unloaded first to give the converter the RAM)
curl -s -m 30 -X POST http://127.0.0.1:9090/api/models/unload/Halocline-QFN-UD-IQ4_XS -o /dev/null
cd "$WT"
if [ -f "$PACK" ]; then log "pack exists, convert skipped: $(stat -c %s "$PACK") bytes"; else
log "convert dry run"
$PY tools/convert/convert.py --gguf "$S1" --sections-from "$UD" --name "QFN-SC117-GSQ-RCO-abliterated-IQ3_S" \
    --out "$PACK" --dry-run > "$O/convert-dry.txt" 2>&1 || die "dry run failed: $(tail -3 "$O/convert-dry.txt" | tr '\n' ' ')"
log "dry run ok"
if [ ! -f "$PACK" ]; then
  log "convert start"
  systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- \
    $PY tools/convert/convert.py --gguf "$S1" --sections-from "$UD" --name "QFN-SC117-GSQ-RCO-abliterated-IQ3_S" \
    --out "$PACK" > "$O/convert.txt" 2>&1 || die "convert failed: $(tail -3 "$O/convert.txt" | tr '\n' ' ')"
fi
log "pack: $(stat -c %s "$PACK") bytes"
fi

# 3. benches under one Berth lease
cat > "$O/bench.sh" <<'EOB'
#!/usr/bin/env bash
set -u
O=$(cd "$(dirname "$0")" && pwd)
. "$O/bench.env"
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }
power() { sudo "$TUNE" --power "$1" --max-clock 2600 --offset 150 >> "$O/gpu-tune.log" 2>&1; log "power limit now $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"; }
watch_thermal() {   # 1 s samples; at hw slowdown or 83 C while above 350 W, back to 350 W and mark the run
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
    --format=csv,noheader -lms 1000 | while IFS=, read -r ts temp pw pl sm hw hwt swt; do
      echo "$ts,$temp,$pw,$pl,$sm,$hw,$hwt,$swt" >> "$O/thermal-$1.csv"
      t=${temp// /}; l=${pl// /}; l=${l%%.*}
      h1=${hw// /}; h2=${hwt// /}
      if [ "${l:-0}" -gt 350 ] && { [ "${t:-0}" -ge 83 ] || [ "$h1" = Active ] || [ "$h2" = Active ]; }; then
        echo "THERMAL $1 $ts temp=$t hw=$hw hwt=$hwt" >> "$O/run.log"; power 350; touch "$O/thermal-$1.tripped"
      fi
    done
}
run() {   # name pack watts
  [ "$3" = 450 ] && power 450
  watch_thermal "$1" & W=$!
  log "$1 bench start: $(basename "$2") at $3 W"
  "$R/bin/halocline-bench" --community --model "$2" --submitter T-Crypt --community-file "$WT/docs/community/benchmarks.json" \
      > "$O/$1.bench.txt" 2>&1
  log "$1 bench rc=$? $( [ -f "$O/thermal-$1.tripped" ] && echo THERMAL-LIMITED )"
  kill $W 2>/dev/null; pkill -P $W 2>/dev/null; pkill -f "nvidia-smi --query-gpu=timestamp" 2>/dev/null
  [ "$3" = 450 ] && power 350
}
run A-ud-350 "$UD" 350
run B-sc117-350 "$PACK" 350
run C-sc117-450 "$PACK" 450
run D-ud-450 "$UD" 450
sudo systemctl restart gpu-tune.service; log "gpu-tune.service restarted: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
EOB
chmod +x "$O/bench.sh"
printf 'R=%q\nUD=%q\nPACK=%q\nWT=%q\nTUNE=%q\n' "$R" "$UD" "$PACK" "$WT" "$TUNE" > "$O/bench.env"
log "bench queue start"
berth train --as halocline-sc117-bench -- "$O/bench.sh" > "$O/berth.log" 2>&1
log "bench queue rc=$?; power limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
log "done"
