#!/usr/bin/env bash
# The dense CPU fallback (halo-cyber): --vision dynamic encodes on the card while its 966 MiB arena fits, and on the CPU
# (halocline-vision's --vision cpu encoder) when it does not. OrcaSAQ-2 vision pack, the served profile (262,144, K8V4,
# --spec 8). Images after load (GPU), a 137K-token prompt (the case that refused images before), the images again
# (expect the CPU fallback, logged), replies compared with the --vision cpu run of ../halocline-orcasaq-vision.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
H=$HOME/wt/halocline-halo-cyber
P=/data/models/halocline/packs/OrcaSAQ-2-27B-Uncensored-vision.halo
VC=$HOME/homelab/state/evals/2026-11-19/halocline-vision/vision_check.py
log() { echo "$(date -Is) $*" | tee -a "$O/run.log.txt"; }
"$H/bin/halocline" serve "$P" --python "$HOME/halocline/releases/b22ecbb2/.venv/bin/python" --host 127.0.0.1 --port 8096 \
    --alias Halocline-OrcaSAQ-27B --spec 8 --max-context 262144 --kv k8v4 --mtp-kv int8 --mtp-window 32768 --vision dynamic \
    --draft-vocab "$H/tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin" > "$O/serve.out" 2>&1 &
S=$!
for t in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:8096/v1/models | grep -q '"loaded"' && break; kill -0 $S || { log "server exited"; exit 1; }; sleep 2; done
# round 4: no images before the long prompt, so the encoder's arena is not already on the card (as through llama-swap)
cp "$HOME/homelab/state/evals/2026-11-20/halocline-orcasaq-fallback/round1/vision-1-after-load.json" "$O/vision-1-after-load.json"; : > "$O/vision-1-after-load.txt"
log "after load: $(grep -c '^PASS' "$O/vision-1-after-load.txt")/3 PASS, cpu-fallback lines so far: $(grep -c 'encoded on the CPU' "$O/serve.out")"
# (round 3: no smoke; the helper process takes the room instead)
: python3 "$HOME/homelab/scripts/llama-swap/lsw.py" smoke Halocline-OrcaSAQ-27B --url http://127.0.0.1:8096 --long > "$O/smoke-${ROUND:-1}.txt" 2>&1
log "smoke (the sequence that refused images through llama-swap): $(grep -c '^PASS' "$O/smoke-${ROUND:-1}.txt")/5 PASS"
python3 - <<'PY' > "$O/long-prompt.txt" 2>&1
import json, random, urllib.request, time
random.seed(1); w = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
text = "\n".join(" ".join(random.choice(w) for _ in range(12)) + "." for _ in range(136000 // 18))
b = {"model": "Halocline-OrcaSAQ-27B", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": text + "\n\nSay OK."}]}
t0 = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8096/v1/chat/completions",
     json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
print("prompt_tokens", r["usage"]["prompt_tokens"], "s", round(time.time() - t0, 1), repr(r["choices"][0]["message"]["content"]))
PY
log "long prompt: $(cat "$O/long-prompt.txt" | tr '\n' ' ' | cut -c1-120) vram $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
KEEP_MIB=200 HOLD_S=120 python3 "$O/hog.py" > "$O/hog.txt" 2>&1 & HOG=$!
sleep 4; log "$(cat "$O/hog.txt")"
python3 "$VC" Halocline-OrcaSAQ-27B --url http://127.0.0.1:8096 --out "$O/vision-2-after-long.json" > "$O/vision-2-after-long.txt" 2>&1
kill -TERM $HOG 2>/dev/null; wait $HOG 2>/dev/null
log "after the long prompt: $(grep -c '^PASS' "$O/vision-2-after-long.txt")/3 PASS, cpu-fallback lines: $(grep -c 'encoded on the CPU' "$O/serve.out")"
kill -TERM $S; wait $S 2>/dev/null
R=$HOME/homelab/state/evals/2026-11-20/halocline-orcasaq-vision/vision-cpu.json
python3 - "$R" "$O/vision-2-after-long.json" "$O/vision-1-after-load.json" <<'PY' | tee -a "$O/run.log.txt"
import json, sys
def replies(p): return {r["case"]: r.get("reply") for r in json.load(open(p))["results"]}
cpu, fb, gpu = (replies(p) for p in sys.argv[1:])
print("fallback replies == --vision cpu replies:", fb == cpu, "| after-load (GPU) == cpu:", gpu == cpu)
for k in cpu: print(" ", k, repr(cpu[k])[:50], "|", repr(fb.get(k))[:50])
PY
log end
