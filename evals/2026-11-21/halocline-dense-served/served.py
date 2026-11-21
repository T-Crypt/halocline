#!/usr/bin/env python3
"""Halocline dense as served: the decode and prefill rates llama-swap recorded for real requests (2026-11-21).

Reads llama-swap's activity log (GET /api/metrics/activity, its tokens.* fields are the engine's own `timings`) for one
model and keeps the requests that came from another machine (the operator's coding agent over the tailnet); local
requests (benches, smoke checks) are dropped. Writes rows.tsv (no addresses, no prompt or reply text) and prints the
summary. `--rows rows.tsv` re-summarises a saved file.

  python3 served.py --url http://127.0.0.1:9090 --model Halocline-27B-UD-Q4_K_XL --since 2026-11-21T09:00 > summary.txt
"""
import argparse, json, statistics as st, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = ["timestamp", "input_tokens", "cache_tokens", "output_tokens", "draft_tokens", "draft_acc_tokens",
        "prompt_per_second", "tokens_per_second", "duration_ms"]


def fetch(url, model, since):
    rows, page = [], 1
    while True:
        d = json.load(urllib.request.urlopen(f"{url}/api/metrics/activity?page={page}&limit=500"))
        rows += d["data"]
        if page >= d["total_pages"]:
            break
        page += 1
    out = []
    for r in rows:
        src = r.get("src", "")
        if r["model"] != model or r["timestamp"] < since or src.endswith(("127.0.0.1", "::1")):
            continue
        t = r["tokens"]
        out.append([r["timestamp"]] + [t.get(c, 0) for c in COLS[1:-1]] + [r["duration_ms"]])
    return sorted(out)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def summary(rows):
    R = [dict(zip(COLS, r)) for r in rows]
    dec = [r for r in R if float(r["tokens_per_second"]) > 0]
    long_out = [r for r in dec if int(r["output_tokens"]) >= 200]
    pre = [r for r in R if int(r["input_tokens"]) >= 1000]
    ctx = [int(r["cache_tokens"]) + int(r["input_tokens"]) for r in R if int(r["cache_tokens"]) + int(r["input_tokens"])]
    cut = len(R) - len(ctx)   # a request ended before its prompt ran (an unload) records no tokens
    acc = [int(r["draft_acc_tokens"]) / int(r["draft_tokens"]) for r in dec if int(r["draft_tokens"])]
    tps = lambda rs: [float(r["tokens_per_second"]) for r in rs]
    print(f"requests {len(R)} ({R[0]['timestamp']} to {R[-1]['timestamp']}; {cut} with no tokens), context {min(ctx):,}-{max(ctx):,} tokens")
    o = sorted(int(r["output_tokens"]) for r in dec)
    print(f"reply tokens: p25 {q(o, .25)}, median {st.median(o):.0f}, p75 {q(o, .75)}, max {o[-1]}")
    for name, rs in [("decode, every reply", dec), ("decode, replies of 200+ tokens", long_out)]:
        v = tps(rs)
        print(f"{name}: n={len(v)} median {st.median(v):.1f} p25 {q(v, .25):.1f} p75 {q(v, .75):.1f} p95 {q(v, .95):.1f} max {max(v):.1f} t/s")
    v = [float(r["prompt_per_second"]) for r in pre]
    print(f"prefill, chunks of 1,000+ new tokens over a warm cache: n={len(v)} median {st.median(v):.0f} max {max(v):.0f} t/s")
    print(f"draft tokens accepted: median {st.median(acc):.0%}, best {max(acc):.0%}")


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:9090")
ap.add_argument("--model", default="Halocline-27B-UD-Q4_K_XL")
ap.add_argument("--since", default="2026-11-21T09:00")
ap.add_argument("--rows")
a = ap.parse_args()
if a.rows:
    rows = [l.rstrip("\n").split("\t") for l in open(a.rows)][1:]
else:
    rows = fetch(a.url, a.model, a.since)
    with open(HERE / "rows.tsv", "w") as f:
        f.write("\t".join(COLS) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
summary(rows)
