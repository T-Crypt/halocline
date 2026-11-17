#!/usr/bin/env bash
cd ~/wt/halocline-import
mkdir -p ~/wt/halocline-tools/dense-engine/fit
for q in UD-IQ4_XS UD-Q4_K_XL; do for kv in int8 fp16; do
  ~/.local/bin/berth run --as halocline-dense-fit --priority foreground --pinned -- ./bin/halo     --model /data/models/halocline/packs/Qwen3.8-27B-$q-vision.basalt --vision --tokens-file ~/wt/halocline-tools/dense-engine/btree.ids --max-new 8 --spec 8     --max-context 262144 --kv $kv > ~/wt/halocline-tools/dense-engine/fit/$q-$kv.out 2> ~/wt/halocline-tools/dense-engine/fit/$q-$kv.err
  echo "$q $kv rc=$?" >> ~/wt/halocline-tools/dense-engine/fit/summary.txt
done; done
echo "queue3 done $(date -Is)"
