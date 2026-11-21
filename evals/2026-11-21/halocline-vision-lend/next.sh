#!/usr/bin/env bash
# After the ABBA run: the thread default in both trees, both builds, then the gates and the after-long check under one
# training lease. Each step logs to pipeline.log; a failed step stops the pipeline.
set -u
export PATH=$HOME/.local/bin:/opt/cuda/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
log() { echo "$(date -Is) $*" | tee -a "$O/pipeline.log"; }
while systemctl --user is-active -q halo-lend-ab; do sleep 10; done
log "ABBA ended: $(tail -1 "$O/berth.log")"
grep -q '^rc=0' "$O/berth.log" || { log "ABBA did not exit 0: stopping"; exit 1; }
# (2026-11-21 10:15: the patch was applied to both trees on the first start; the configure then failed on PATH)
git -C $HOME/wt/halocline-vision-arena diff --quiet -- serve/engine/vision.py && { log "patch missing"; exit 1; }
B=$HOME/wt/halocline-vision-arena
( cd $HOME/wt/halocline-orca-lend && cmake -S . -B build-89 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON > cmake.log 2>&1 ) || { log "orca cmake failed"; exit 1; }
for W in $B $HOME/wt/halocline-orca-lend; do
  ( cd $W && flock /tmp/halocline-build.lock nice cmake --build build-89 -j 24 >> build.log 2>&1 ); rc=$?
  log "build $W rc=$rc"; [ $rc = 0 ] || exit 1
done
berth train --as halocline-vision-gates -- bash -c "$O/gates.sh && $O/after-long.sh" >> "$O/pipeline.log" 2>&1
log "gates + after-long rc=$?"
