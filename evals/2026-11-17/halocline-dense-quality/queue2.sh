#!/usr/bin/env bash
while kill -0 958375 2>/dev/null; do sleep 5; done
cd ~/wt/halocline-tools/dense-engine
./llama-kld.sh UD-IQ4_XS ~/wt/halocline-tools/dense-engine/d7/llama-iq4xs-f16
./llama-kld.sh UD-IQ4_XS ~/wt/halocline-tools/dense-engine/d7/llama-iq4xs-q8 q8_0 q8_0
echo "queue2 done $(date -Is)"
