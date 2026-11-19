#!/usr/bin/env python3
"""A needle near the start of a ~N-token prompt (the smoke's filler), asked at the end: does the long profile hold and
read 250K tokens. needle.py N OUT.json"""
import json, random, sys, time, urllib.request
N = int(sys.argv[1])
random.seed(7)
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
lines = [" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(N // 18)]
lines.insert(len(lines) // 50, "Remember this: the vault code is 48213.")
content = "\n".join(lines) + "\n\nWhat is the vault code given in the text above? Answer with the number only."
model = json.load(urllib.request.urlopen("http://127.0.0.1:8096/v1/models"))["data"][0]["id"]
body = {"model": model, "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}]}
t0 = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8096/v1/chat/completions",
              json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
ans = d["choices"][0]["message"].get("content") or ""
row = {"needle_tokens": d["usage"]["prompt_tokens"], "wall_s": round(time.time() - t0, 1), "answer": ans.strip()[:40],
       "found": "48213" in ans, "timings": d.get("timings")}
print(json.dumps(row)); json.dump(row, open(sys.argv[2], "w"), indent=1)
