#!/usr/bin/env python3
"""Window-weighted stage shares per workload from out/*.profile.txt (W13 --profile tables).

    shares.py   -> markdown: per workload, mean window ms, mean rows T, and each stage's share of the window
"""
import re
from pathlib import Path
O = Path(__file__).resolve().parent / "out"
ORDER = ["short-prose", "short-struct", "d36k-prose", "d36k-struct", "d137k-prose", "d137k-struct"]
GROUPS = {"attention (pre+qk+core+out)": ["attn-pre", "attn-qk", "attn-core", "attn-out"],
          "attn-core (KV read)": ["attn-core"], "GDN (pre+rec+out)": ["gdn-pre", "gdn-rec", "gdn-out"],
          "FFN (up+down)": ["ffn-up", "ffn-down"], "LM head": ["head"], "MTP drafter (draft+hist)": ["mtp-draft", "mtp-hist"],
          "other (embed, sample, gap)": ["embed", "sample", "gap"]}
row = re.compile(r"T (\d+) windows (\d+) window ([\d.]+) ms \| (.*)$")
print("| workload | windows | mean T | window ms | " + " | ".join(GROUPS) + " |")
print("|---" * (4 + len(GROUPS)) + "|")
for name in ORDER:
    p = O / f"{name}.profile.txt"
    if not p.exists(): continue
    n = tw = tt = 0.0; st = {}
    for line in p.read_text().splitlines():
        m = row.search(line)
        if not m: continue
        T, w, ms = int(m[1]), int(m[2]), float(m[3])
        n += w; tw += w * ms; tt += w * T
        for part in m[4].split(" | "):
            q = re.match(r"([\w-]+) ([\d.]+)", part.replace("stages ", "stages ").strip())
            if q and q[1] != "stages": st[q[1]] = st.get(q[1], 0.0) + w * float(q[2])
        g = re.search(r"gap ([\d.]+)", m[4])
        if g: st["gap"] = st.get("gap", 0.0) + w * float(g[1])
    cells = [f"{100 * sum(st.get(s, 0) for s in v) / tw:.1f}%" for v in GROUPS.values()]
    print(f"| {name} | {int(n)} | {tt / n:.2f} | {tw / n:.2f} | " + " | ".join(cells) + " |")
