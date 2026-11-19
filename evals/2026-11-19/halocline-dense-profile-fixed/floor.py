#!/usr/bin/env python3
"""Per stage: the last profile table's mean ms at each fixed width against the stage's weight-read floor
(stage_bytes.py's bytes / 1,008 GB/s, the 4090's spec bandwidth). Uncaptured, PDL off, so ms include launch gaps.
floor.py OUTDIR"""
import re, sys, json
from pathlib import Path
MiB = 2**20
BYTES = {"ffn-up": 6822.6, "ffn-down": 3550.1, "gdn-pre": 2460.0, "gdn-out": 1052.6, "head": 994.6, "attn-pre": 800.8, "attn-out": 393.0}
BW = 1008e9
floor = {k: v * MiB / BW * 1e3 for k, v in BYTES.items()}
out = Path(sys.argv[1])
for f in sorted(out.glob("*.profile.txt"), key=lambda p: (p.stem.split('-')[1], int(p.stem[1:].split('-')[0]))):
    tabs = {}
    for line in f.read_text().splitlines():
        m = re.search(r"T (\d+) windows (\d+) window ([\d.]+) ms(.*)", line)
        if not m: continue
        st = {k: float(v) for k, v in re.findall(r"\| ([\w-]+) ([\d.]+) \(", m.group(4))}
        gap = re.search(r"gap (-?[\d.]+)", m.group(4))
        tabs[int(m.group(1))] = (int(m.group(2)), float(m.group(3)), st, float(gap.group(1)) if gap else 0)
    for T, (n, wm, st, gap) in sorted(tabs.items()):
        if n < 50: continue
        print(f"{f.stem:10s} T{T} windows {n} window {wm:.2f} ms  gap {gap:.2f}")
        for k in ["ffn-up", "ffn-down", "gdn-pre", "gdn-rec", "gdn-out", "attn-pre", "attn-qk", "attn-core", "attn-out", "head", "embed", "sample", "mtp-draft", "mtp-hist"]:
            if k not in st: continue
            fl = floor.get(k)
            extra = f" floor {fl:5.2f}  over {st[k]-fl:+5.2f} ms ({100*fl/st[k]:3.0f}% of BW)" if fl else ""
            print(f"    {k:9s} {st[k]:6.2f}{extra}")
