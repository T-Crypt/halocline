#!/usr/bin/env bash
# Does the halo-cyber loader change leave a UD pack's output alone? UD-Q4_K_XL (Q8_0 ssm_alpha/beta), release b2f81178
# against the halo-cyber build, --spec 0 (one row a window: no timing-dependent drafts), greedy, short prose + JSON.
set -u
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin TMPDIR=/var/tmp/halocline
O=$(cd "$(dirname "$0")" && pwd)
P=/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo
for arm in ${ARMS:-release cyber}; do
  case $arm in release) B=$HOME/halocline/releases/b2f81178 ;; cyber) B=$HOME/wt/halocline-halo-cyber ;; esac
  "$B/bin/halocline" serve "$P" --python "$HOME/halocline/releases/b22ecbb2/.venv/bin/python" --host 127.0.0.1 --port 8096 \
      --alias ud --spec 0 --max-context 32768 --kv k8v4 --vision cpu > "$O/ud-$arm.serve.out" 2>&1 &
  S=$!
  for t in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:8096/v1/models | grep -q '"loaded"' && break; kill -0 $S || break; sleep 2; done
  python3 - "$O/ud-$arm.txt" <<'PY'
import json, sys, urllib.request, hashlib
out = []
for p in ["Write a long story, about 300 words, about a lighthouse keeper.", "Output only a JSON array of 10 objects with id, name and city."]:
    b = {"model": "ud", "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": p}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8096/v1/chat/completions", json.dumps(b).encode(),
                                                                {"Content-Type": "application/json"}), timeout=600))
    t = r["choices"][0]["message"]["content"]; out.append(hashlib.sha256(t.encode()).hexdigest()[:16] + " " + str(len(t)))
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY
  kill -TERM $S; wait $S 2>/dev/null; sleep 4
done
diff -q "$O/ud-release.txt" "$O/ud-cyber.txt" && echo "IDENTICAL" > "$O/ud-identity.result" || echo "DIFFER" > "$O/ud-identity.result"
cat "$O/ud-release.txt" "$O/ud-cyber.txt" >> "$O/ud-identity.result"
