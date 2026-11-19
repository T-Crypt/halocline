#!/usr/bin/env bash
# Sparse depth speed: is the 2026-11-19 drop the RAM reserve or the pack's page cache? (ticket SRT2J9G)
# Run under: berth train --as halocline-sparse-abba -- run.sh [ARM ...]
# Arms (release 5aec8e3a, served sparse flags, :8096, lsw.py smoke --long):
#   r2048  vision pack, no --vision, --ram-reserve-mib 2048   (last night's RAM plan on today's file)
#   r4096  vision pack, --vision cpu, --ram-reserve-mib 4096  (served today)
#   nvme   as r4096, the pack on /mnt/models (PCIe NVMe) instead of /data (SATA SSD, shared with /)
#   old    old pack (no projector), --ram-reserve-mib 2048    (last night's exact command)
# Default order: r2048 r4096 r4096 r2048 old. The text sections of both packs are byte-identical
# (../halocline-vision/sections-old-vs-vision.txt), so r2048 vs old isolates the file's page cache.
# contended/: the first try, stopped: DeepSeek workers ran the server test suite on all cores (load 18). Re-run with
# the workers pinned to E-cores 28-31 (taskset).
# Holds /tmp/halocline-build.lock for the whole run, so no worker build runs beside the pinned expert tier.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
R=$HOME/halocline/releases/5aec8e3a
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
NEW=/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
OLD=/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.halo
NVME=/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo   # the same file copied to the NVMe (cmp checked)
export TMPDIR=/var/tmp/halocline
log() { echo "$(date -Is) $*" | tee -a "$O/run.log"; }

exec 9>/tmp/halocline-build.lock
flock 9
log "build lock held; load $(cut -d" " -f1-3 /proc/loadavg); worker affinity $(for p in $(pgrep -f "opencode run -m opencode-go"); do taskset -cp $p | cut -d: -f2; done | sort -u | tr "\n" " "); power: $(nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader)"

tree_pids() { local p=$1; echo "$p"; for c in $(pgrep -P "$p"); do tree_pids "$c"; done; }
tree_read() { local s=0 b; for p in $(tree_pids "$1"); do b=$(awk '/^read_bytes/{print $2}' /proc/$p/io 2>/dev/null); s=$((s + ${b:-0})); done; echo $s; }

n=0
for arm in ${*:-r2048 r4096 r4096 r2048 old}; do
  n=$((n + 1)); tag=$n-$arm
  case $arm in
    r2048) pack=$NEW; extra="--ram-reserve-mib 2048" ;;
    r4096) pack=$NEW; extra="--vision cpu --ram-reserve-mib 4096" ;;
    old)   pack=$OLD; extra="--ram-reserve-mib 2048" ;;
    nvme)  pack=$NVME; extra="--vision cpu --ram-reserve-mib 4096" ;;   # as r4096, pack on the NVMe
    nvme2048) pack=$NVME; extra="--ram-reserve-mib 2048" ;;
    *) log "unknown arm $arm"; continue ;;
  esac
  log "$tag start: $(basename $pack) $extra; MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB, cached $(awk '/^Cached/{print int($2/1024)}' /proc/meminfo) MiB"
  "$R/bin/halocline" serve "$pack" --python "$PY" --host 127.0.0.1 --port 8096 --alias sparse-abba \
      --spec 8 --prefill auto --max-context 262144 --kv int8 $extra > "$O/$tag.serve.out" 2>&1 &
  SRV=$!
  ( while kill -0 $SRV 2>/dev/null; do
      echo "$(date +%T) avail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) cached_mib=$(awk '/^Cached/{print int($2/1024)}' /proc/meminfo) read_mib=$(( $(tree_read $SRV) >> 20 ))"
      sleep 2; done ) > "$O/$tag.ram.log" 2>&1 &
  TR=$!
  for t in $(seq 1 900); do grep -q "weights  loaded" "$O/$tag.serve.out" && break; kill -0 $SRV 2>/dev/null || break; sleep 1; done
  r0=$(tree_read $SRV)
  python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke sparse-abba --url http://127.0.0.1:8096 --long \
      --out "$O/$tag.smoke.json" > "$O/$tag.smoke.txt" 2>&1
  rc=$?
  r1=$(tree_read $SRV)
  log "$tag smoke rc=$rc read_during_smoke_mib=$(( (r1 - r0) >> 20 )) ram_tier=$(grep 'system ram' "$O/$tag.serve.out" | tail -1 | grep -oE '[0-9.]+/[0-9.]+ GiB')"
  grep -E '^(PASS|FAIL)' "$O/$tag.smoke.txt" | sed "s/^/$tag  /" | tee -a "$O/run.log"
  kill -TERM $SRV; wait $SRV 2>/dev/null; kill $TR 2>/dev/null
  sleep 5
done
log "done"
