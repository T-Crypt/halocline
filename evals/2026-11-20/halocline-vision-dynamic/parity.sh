#!/usr/bin/env bash
# CUDA encoder vs CPU encoder (same bf16 input rounding), per tap, on the 19 airbench images (outside the repo).
set -u
B=$HOME/halocline/releases/aa156951/build-89/tools/vision/vision_cuda_parity
PACK=/data/models/halocline/vision/unsloth-qfn-814db05f/mmproj-BF16.gguf   # the projector the QFN pack carries
O=$(cd "$(dirname "$0")" && pwd)
for f in /var/tmp/vision-ab/img/*.png; do
  k=$(basename $f .png)
  echo "== $k"
  $B encoder $PACK $f 2>&1  | grep -vE '^ *$'
done > "$O/parity.txt"
echo "exit done"
