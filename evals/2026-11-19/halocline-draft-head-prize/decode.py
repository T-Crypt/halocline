#!/usr/bin/env python3
"""The six vs_ninfer workloads, thinking off, direct to :8096, streamed, greedy, 1,200 tokens: client decode rate
(tokens - 1) / (last delta - first delta), as ../halocline-vs-ninfer-nothink measures both engines. decode.py OUT.json"""
import hashlib, json, random, sys, time, urllib.request
URL = "http://127.0.0.1:8096/v1/chat/completions"
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
STRUCT = ("Output only a JSON array of 40 objects, one per fictional employee, each with the fields id (integer), name, "
          "city, department, salary (integer), start_date (YYYY-MM-DD) and skills (an array of three strings).")
WORK = [("short-prose", 0, PROSE), ("short-struct", 0, STRUCT), ("d36k-prose", 36000, PROSE),
        ("d36k-struct", 36000, STRUCT), ("d137k-prose", 136000, PROSE), ("d137k-struct", 136000, STRUCT)]
def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return "\n".join(" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(tokens // 18))
model = json.load(urllib.request.urlopen("http://127.0.0.1:8096/v1/models"))["data"][0]["id"]
res = []
for name, depth, task in WORK:
    content = (filler(depth) + "\n\nIgnore the lines above. " if depth else "") + task
    body = {"model": model, "max_tokens": 1200, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    t0 = time.time(); first = last = None; n = 0; usage = None; text = []; tm = None
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            if d.get("timings"): tm = d["timings"]
            for c in d.get("choices", []):
                s = (c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning_content") or "")
                if s:
                    now = time.time(); first = first or now; last = now; n += 1; text.append(s)
    toks = (usage or {}).get("completion_tokens", n)
    row = {"work": name, "ttft_s": round(first - t0, 2) if first else None, "completion_tokens": toks,
           "decode_tps": round((toks - 1) / (last - first), 1) if first and last > first else None,
           "drafts": (tm or {}).get("draft_n_accepted"), "offered": (tm or {}).get("draft_n"), "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}
    print(json.dumps(row), flush=True); res.append(row)
json.dump(res, open(sys.argv[1], "w"), indent=1)
