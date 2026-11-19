#!/usr/bin/env bash
# The two dense profiles as a repo user runs them: `serve --config examples/<profile>.json` from the branch tree, one at
# a time on :8096, each torn down before the next. Per profile: the engine's own report (context, KV, lossy flags),
# lsw.py smoke --long, the six decode workloads with thinking off (corners-style client rate), and for the long
# profile a needle at ~250K tokens. Run under: berth train --as halocline-profiles -- profiles.sh
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
B=$HOME/wt/halocline-long-ctx
PY=$HOME/halocline/releases/b22ecbb2/.venv/bin/python
MODELS=/data/models/halocline/packs
one() {  # $1 profile name (examples/$1.json)
  local p=$1 cfg=$O/$1.config.json
  # the example as shipped, with the model path pointed at this machine's pack and the test port
  python3 - "$B/examples/$p.json" "$cfg" "$MODELS" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["server"]["port"] = 8096
for m in c["models"]: m["model"] = sys.argv[3] + "/" + m["model"].split("/")[-1]
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  # exactly as the guide says: bin/halocline serve --config FILE (plus --python for this machine's venv)
  "$B/bin/halocline" serve --config "$cfg" --python "$PY" > "$O/$p.serve.out" 2>&1 &
  local SRV=$! s=""
  for t in $(seq 1 600); do
    s=$(curl -s -m 5 http://127.0.0.1:8096/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['status']['value'])" 2>/dev/null)
    [ "$s" = loaded ] && break; sleep 2
  done
  echo "$(date -Is) $p: $s" >> "$O/profiles.log"
  curl -s http://127.0.0.1:8096/v1/models > "$O/$p.models.json"
  curl -s http://127.0.0.1:8096/v1/status > "$O/$p.status.json"
  grep -E "lossy|long-context|kv=|ctx=" "$O/$p.serve.out" | head -20 >> "$O/profiles.log"
  python3 $HOME/homelab/scripts/llama-swap/lsw.py smoke "$p" --url http://127.0.0.1:8096 --long --out "$O/$p.smoke.json" > "$O/$p.smoke.txt" 2>&1
  echo "smoke rc=$?" >> "$O/$p.smoke.txt"
  python3 "$O/decode.py" "$O/$p.decode.json" >> "$O/profiles.log" 2>&1
  [ "$p" = qwen38-27b ] && python3 "$O/needle.py" 250000 "$O/$p.needle.json" >> "$O/profiles.log" 2>&1
  kill -TERM $SRV; wait $SRV
  sleep 3   # the wrapper forwards the stop to the server and engine: nothing of this profile may stay on the card
  echo "$(date -Is) $p stopped; on the card: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')" >> "$O/profiles.log"
}
one qwen38-27b-k8v4
one qwen38-27b
echo "$(date -Is) end" >> "$O/profiles.log"
