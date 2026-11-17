#!/usr/bin/env bash
# D7 llama.cpp arm: the same Unsloth GGUF against the same 27B BF16 base, 100 chunks. Usage: llama-kld.sh QUANT OUTDIR [ctk ctv]
set -u
Q=$1; OUT=$2; CTK=${3:-f16}; CTV=${4:-f16}
G=/mnt/models/Ollama/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-$Q.gguf
BASE=/data/models/halocline/kld-bases/qwen38-27b-bf16-base.kld
mkdir -p "$OUT"
~/.local/bin/berth run --as llama-kld --priority foreground --pinned -- \
  ~/wt/llama.cpp-2cbe0ec/build/bin/llama-perplexity -m "$G" --kl-divergence-base "$BASE" --kl-divergence \
    --chunks 100 -c 512 -b 512 -ub 512 -ngl 99 -fa on -ctk "$CTK" -ctv "$CTV" > "$OUT/kld.txt" 2> "$OUT/kld.err"
echo "rc=$?" >> "$OUT/kld.txt"
