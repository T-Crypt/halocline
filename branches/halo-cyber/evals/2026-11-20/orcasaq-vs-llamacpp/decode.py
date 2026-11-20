#!/usr/bin/env python3
"""The six vs_ninfer workloads (../../2026-11-19/halocline-w35-confidence-ab/decode.py), thinking off, greedy, streamed,
1,200 tokens, plus needles: one passphrase at 25% depth of N filler tokens. decode.py URL OUT.json [NEEDLE_DEPTH ...]
Client decode rate = (tokens - 1) / (last delta - first delta), the same for both engines."""
import hashlib, json, random, sys, time, urllib.request
BASE, OUT, NEEDLES = sys.argv[1].rstrip("/"), sys.argv[2], [int(x) for x in sys.argv[3:]]
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
STRUCT = ("Output only a JSON array of 40 objects, one per fictional employee, each with the fields id (integer), name, "
          "city, department, salary (integer), start_date (YYYY-MM-DD) and skills (an array of three strings).")
WORK = [("short-prose", 0, PROSE), ("short-struct", 0, STRUCT), ("d36k-prose", 36000, PROSE),
        ("d36k-struct", 36000, STRUCT), ("d137k-prose", 136000, PROSE), ("d137k-struct", 136000, STRUCT)]
CODE = "7-3-9-1-4"
def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return ["%s." % " ".join(random.choice(words) for _ in range(12)) for _ in range(tokens // 18)]
model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
def run(name, content, max_tokens=1200):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    t0 = time.time(); first = last = None; n = 0; usage = None; text = []
    with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                       {"Content-Type": "application/json"}), timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                s = (c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning_content") or "")
                if s:
                    now = time.time(); first = first or now; last = now; n += 1; text.append(s)
    toks = (usage or {}).get("completion_tokens", n); out = "".join(text)
    return {"work": name, "prompt_tokens": (usage or {}).get("prompt_tokens"), "ttft_s": round(first - t0, 2) if first else None,
            "completion_tokens": toks, "decode_tps": round((toks - 1) / (last - first), 1) if first and last > first else None,
            "text_sha": hashlib.sha256(out.encode()).hexdigest()[:16]}, out
res = []
for name, depth, task in WORK:
    row, _ = run(name, ("\n".join(filler(depth)) + "\n\nIgnore the lines above. " if depth else "") + task)
    print(json.dumps(row), flush=True); res.append(row)
for depth in NEEDLES:
    lines = filler(depth); lines.insert(len(lines) // 4, "The vault code is %s. Remember it." % CODE)
    row, out = run("needle-%dk" % (depth // 1000), "\n".join(lines) + "\n\nWhat is the vault code? Answer with the code only.", 32)
    row["found"] = CODE.replace("-", "") in "".join(ch for ch in out if ch.isdigit()); row["reply"] = out.strip()[:40]
    print(json.dumps(row), flush=True); res.append(row)
json.dump(res, open(OUT, "w"), indent=1)
