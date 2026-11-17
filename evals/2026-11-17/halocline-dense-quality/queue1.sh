#!/usr/bin/env bash
while kill -0 948788 2>/dev/null; do sleep 5; done
cd ~/wt/halocline-tools/dense-engine
./d4.sh /data/models/halocline/packs/Qwen3.8-27B-UD-IQ4_XS.basalt ~/wt/halocline-tools/dense-engine/d4/iq4xs-int8-mmvq int8
./llama-kld.sh UD-IQ4_XS ~/wt/halocline-tools/dense-engine/d7/llama-iq4xs-f16
./d4.sh /data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt ~/wt/halocline-tools/dense-engine/d4/q4kxl-int8-mmvq int8
./llama-kld.sh UD-Q4_K_XL ~/wt/halocline-tools/dense-engine/d7/llama-q4kxl-f16
echo "queue1 done $(date -Is)"
