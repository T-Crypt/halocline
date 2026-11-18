#!/usr/bin/env python3
"""THINKING OFF (chat_template_kwargs enable_thinking false) on both engines, so neither splits its reply into a
thinking pass and an answer, and both write the same kind of text; per reply the text hash and the share of reasoning
deltas are recorded (a nonzero share means the engine ignored the switch). Otherwise as the 2026-11-18 run:
Halocline dense against NInfer through llama-swap, the same prompts (ab.py's six workloads: prose and struct at short,
~36K and ~137K tokens of the smoke's needle filler), greedy, 1,200 tokens, streamed. Decode rate is measured the same
way for both engines: (tokens streamed - 1) / (time of the last delta - time of the first), every delta counted
(reasoning and content). Order ABBA over the engines; llama-swap swaps them (Berth drains), so each arm starts cold.

    vs_ninfer.py OUT.json
"""
import hashlib, json, random, sys, time, urllib.request
URL = "http://127.0.0.1:9090/v1/chat/completions"
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
STRUCT = ("Output only a JSON array of 40 objects, one per fictional employee, each with the fields id (integer), name, "
          "city, department, salary (integer), start_date (YYYY-MM-DD) and skills (an array of three strings).")
WORK = [("short-prose", 0, PROSE), ("short-struct", 0, STRUCT), ("d36k-prose", 36000, PROSE),
        ("d36k-struct", 36000, STRUCT), ("d137k-prose", 136000, PROSE), ("d137k-struct", 136000, STRUCT)]

def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return "\n".join(" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(tokens // 18))

def run(model, depth, task):
    content = (filler(depth) + "\n\nIgnore the lines above. " if depth else "") + task
    body = {"model": model, "max_tokens": 1200, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = 0; usage = None; text = []; reasoning = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now; n += 1
                    reasoning += bool(delta.get("reasoning_content"))
                    text.append((delta.get("reasoning_content") or "") + (delta.get("content") or ""))
    toks = (usage or {}).get("completion_tokens", n)
    return {"ttft_s": round(first - t0, 2) if first else None, "deltas": n, "completion_tokens": toks,
            "decode_tps": round((toks - 1) / (last - first), 1) if first and last > first else None,
            "reasoning_share": round(reasoning / n, 3) if n else None,
            "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16], "head": "".join(text)[:80]}

res = []
for model in ["Halocline-27B-UD-Q4_K_XL", "NInfer-Qwen3.8-27B", "NInfer-Qwen3.8-27B", "Halocline-27B-UD-Q4_K_XL"]:
    for name, depth, task in WORK:
        r = {"model": model, "work": name, **run(model, depth, task)}
        print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(sys.argv[1], "w"), indent=1)
