#!/usr/bin/env bash
# After arm 2 of the clean A/B: stop the A/B, copy the vision pack to the NVMe (O_DIRECT, so the copy does not
# seed the page cache), byte-compare, then run arms nvme r4096 r4096 nvme under Berth.
set -u
O=$(cd "$(dirname "$0")" && pwd)
SRC=/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
DST=/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo
until grep -q "2-r4096 smoke rc" "$O/run.log"; do sleep 3; done
sleep 2
pkill -f "halocline-sparse-abba/run.sh|./run.sh" ; sleep 2
for p in $(pgrep -f "alias sparse-abba"); do kill -TERM $p; done
for t in $(seq 1 60); do pgrep -f "alias sparse-abba" >/dev/null || break; sleep 2; done
echo "$(date -Is) A/B stopped after arm 2; copying" | tee -a "$O/nvme.log"
mkdir -p "$(dirname $DST)"
t0=$(date +%s)
dd if="$SRC" of="$DST.part" bs=64M iflag=direct oflag=direct status=none && mv "$DST.part" "$DST"
echo "$(date -Is) copy rc=$? $(( $(date +%s) - t0 )) s" | tee -a "$O/nvme.log"
cmp "$SRC" "$DST" && echo "$(date -Is) cmp identical" | tee -a "$O/nvme.log" || { echo "$(date -Is) cmp DIFFERS" | tee -a "$O/nvme.log"; exit 1; }
mkdir -p "$O/nvme" && cp "$O/run.sh" "$O/nvme/run.sh" && cd "$O/nvme" && berth train --as halocline-sparse-nvme -- ./run.sh nvme r4096 r4096 nvme >> "$O/nvme/berth.log" 2>&1
echo "$(date -Is) nvme arms done" | tee -a "$O/nvme.log"
