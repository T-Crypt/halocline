#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin
# The drafter ring fix (operator 2026-11-17 21:15: "fix the ring first when the card frees up"), run at the head of
# window 6 under its Berth lease (window6.sh calls this first; the prompt-l2 A/B follows whatever happens here).
# Bug: gqa_decode_attn swept a ring from chunk 0, so past position cap a window read no cell (NaN drafter logits):
# int8-140288 with --mtp-window 32768 failed at chunk 0 ("the drafter failed", 21:04). Fix 94ec6c75 (branch
# mtp-ring-fix, ~/wt/halocline-mtp-ring-fix) and its cherry-pick 17ff56ee on d8-q4 (~/wt/halocline-d8-q4-ring).
# Steps, each gated on the one before: 1. gqa_attn_parity (the ring case is new) on both builds; 2. qwen35_mtp_cache_test
# (36K, the 27B pack); 3. a 2-chunk repro at 140,288 with the ring; 4. the full arms the ring blocked: q4-261119 (the D8
# decision: K8V4 ~175K vs Q4 262K) then int8-140288. Same arguments as run.sh; own dirs (*-ring) so nothing is overwritten.
set -u
O=$(cd "$(dirname "$0")" && pwd)
WF=$HOME/wt/halocline-mtp-ring-fix
WQ=$HOME/wt/halocline-d8-q4-ring
PACK=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.basalt
D=/data/models/halocline/kld-bases/depth
base() { echo "$D/qwen38-27b-bf16-depth$1-w1024-c16.kld"; }
log() { echo "$(date -Is) ring: $*" | tee -a "$O/run.log"; }
ping() { notify-phone --level "$1" --title "Halocline drafter ring" "aphotic: $2"; }
q() { python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; print("kld %.5f top1 %.2f ppl %.4f base %.4f rows %d err %s" % (q["kld_mean"], q["top1_pct"], q["ppl"], q["base_ppl"], q["rows"], q.get("error") or "-"))' "$1" 2>/dev/null; }
kld_ok() { python3 -c 'import json,sys; q=json.load(open(sys.argv[1]))["quality"]; sys.exit(0 if q["ran"] and q["kld_mean"] > 0 else 1)' "$1" 2>/dev/null; }
arm() {   # name build depth chunks engine-args...
  local name=$1 bw=$2 depth=$3 chunks=$4; shift 4
  mkdir -p "$O/$name"
  log "arm $name depth $depth chunks $chunks build $(git -C "$bw" rev-parse --short HEAD): $*"
  "$bw/bin/halocline-bench" --model "$PACK" --quality "$chunks" --kld-base "$(base "$depth")" --logs "$O/$name" -o json -- \
      --spec 8 --max-context $((depth + 1033)) "$@" > "$O/$name/bench.json" 2> "$O/$name/bench.err"
  local rc=$?
  log "arm $name rc=$rc $(q "$O/$name/bench.json")"
  [ $rc -eq 0 ] && kld_ok "$O/$name/bench.json"
}

log "start, fix $(git -C "$WF" rev-parse --short HEAD), q4 fix $(git -C "$WQ" rev-parse --short HEAD)"
for b in "$WF" "$WQ"; do
  "$b/build/gqa_attn_parity" > "$O/ring-parity-$(basename "$b").log" 2>&1
  rc=$?; log "gqa_attn_parity $(basename "$b") rc=$rc $(grep 'a ring of' "$O/ring-parity-$(basename "$b").log")"
  [ $rc -eq 0 ] || { ping alert "gqa_attn_parity failed on $(basename "$b") (rc $rc); reruns skipped, prompt-l2 goes on"; exit 1; }
done
"$WF/build/qwen35_mtp_cache_test" "$PACK" > "$O/ring-mtp-cache-test.log" 2>&1
rc=$?; log "qwen35_mtp_cache_test rc=$rc $(tail -1 "$O/ring-mtp-cache-test.log")"
[ $rc -eq 0 ] || { ping alert "qwen35_mtp_cache_test failed (rc $rc); reruns skipped"; exit 1; }
arm int8-140288-ring-repro "$WF" 140288 2 --kv int8 --mtp-kv int8 --mtp-window 32768 ||
  { ping alert "the 140,288 repro still fails with the fix: $(q "$O/int8-140288-ring-repro/bench.json")"; exit 1; }
ping info "fixed: tests pass and the 140,288 ring repro runs ($(q "$O/int8-140288-ring-repro/bench.json" | cut -d' ' -f1-4)); full reruns next"
B=$(base 261119)
if [[ -s $B ]] && python3 -I $HOME/wt/halocline-import/tools/depth_kld/depth_kld.py check "$B" > /dev/null 2>&1; then
  arm q4-261119-ring "$WQ" 261119 16 --kv q4_0 --mtp-kv int8 --mtp-window 32768
  ping $([ $? -eq 0 ] && echo done || echo alert) "q4-261119 (D8 deciding arm): $(q "$O/q4-261119-ring/bench.json")"
else
  log "no checked 261,119 base: q4-261119-ring skipped"; ping alert "no checked 261,119 base, q4-261119 skipped"
fi
arm int8-140288-ring "$WF" 140288 16 --kv int8 --mtp-kv int8 --mtp-window 32768
ping $([ $? -eq 0 ] && echo info || echo alert) "int8-140288: $(q "$O/int8-140288-ring/bench.json")"
log "end"
