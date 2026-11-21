#!/usr/bin/env bash
# One llama-perplexity run under a Berth foreground lease (Unsloth llama.cpp b12588-mix-c0c3de8 + LLAMA_KL_FIRST patch).
# Usage: kld.sh NAME QUANT base|score BASEFILE CTX CHUNKS FIRST(0=stock window) CTK CTV CORPUS
# base  = write BASEFILE from QUANT;  score = score QUANT against BASEFILE.
# --fit on moves tensors off the card as needed; --fit-ctx pins n_ctx so fit never shrinks the context.
set -u
NAME=$1 Q=$2 MODE=$3 BASE=$4 CTX=$5 CH=$6 FIRST=$7 CTK=$8 CTV=$9 CORPUS=${10}
BIN=~/wt/llama.cpp-unsloth-c0c3de8-src/src/build-cuda/bin/llama-perplexity
M=<models>/ae9327c5/Inkling-2-Mini-$Q.gguf
OUT=$(dirname "$0")/$NAME; mkdir -p "$OUT"
B=512; UB=512; [ "$CTX" -gt 4096 ] && B=4096  # ub 512: Inkling relative-bias attention compute buffer OOMs at ub 4096 past ~36K
if [ "$MODE" = base ]; then KL=(--kl-divergence-base "$BASE"); else KL=(--kl-divergence-base "$BASE" --kl-divergence); fi
ENV=(); [ "$FIRST" != 0 ] && ENV=(env LLAMA_KL_FIRST=$FIRST)
echo "$(date -Is) start $NAME $Q $MODE ctx=$CTX chunks=$CH first=$FIRST kv=$CTK/$CTV" >> "$OUT/../queue.log"
T0=$(date +%s)
~/.local/bin/berth run --as inkling-kld --priority foreground --pinned -- \
  "${ENV[@]}" "$BIN" -m "$M" -f "$CORPUS" "${KL[@]}" --chunks "$CH" -c "$CTX" -b $B -ub $UB -fa on \
  -ctk "$CTK" -ctv "$CTV" -t 16 --fit on --fit-ctx "$CTX" > "$OUT/kld.txt" 2> "$OUT/kld.err"
RC=$?
# llama-perplexity exits 0 after a failed decode: success only with its result line
if [ $RC = 0 ] && ! grep -q -E "Mean +KLD|Final estimate" "$OUT/kld.txt" "$OUT/kld.err"; then RC=99; fi
echo "rc=$RC secs=$(( $(date +%s) - T0 ))" >> "$OUT/kld.txt"
echo "$(date -Is) end $NAME rc=$RC secs=$(( $(date +%s) - T0 ))" >> "$OUT/../queue.log"
exit $RC
