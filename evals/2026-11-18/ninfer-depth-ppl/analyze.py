#!/usr/bin/env python3
"""Target-token NLL against the BF16 base on the depth rows: NInfer (window 2 of each prep.py stream) and Halocline
(rows.tsv of the same base: col 3 the engine's NLL, col 4 the BF16 base's NLL of the same target). Per chunk dNLL =
mean NLL - mean base NLL (log of the PPL ratio); mean over 16 chunks with a 95% t interval, and NInfer - Halocline
paired per chunk. Lower dNLL is closer to BF16 on the observed token; it is not KLD (over- and under-confidence can
cancel), and NInfer exposes no logits for KLD.

    analyze.py   (reads this folder and ../../2026-11-17/halocline-d8-depth: the same layout in homelab state/evals
                  and in the Halocline repo's evals/, where rows.tsv is stored as rows.tsv.xz)
"""
import json, lzma, math, statistics as st
from pathlib import Path
O = Path(__file__).resolve().parent
H = O.parent.parent / "2026-11-17" / "halocline-d8-depth"

def rows_text(arm):
    p = H / arm / "rows.tsv"
    return p.read_text() if p.exists() else lzma.open(p.with_suffix(".tsv.xz"), "rt").read()
T95 = 2.131  # t(0.975, df 15)

def halo(arm):
    per = {}
    for line in rows_text(arm).splitlines():
        c = line.split("\t")
        if len(c) >= 10:
            per.setdefault(int(c[0]), []).append((float(c[3]), float(c[4])))
    assert all(len(v) == 1024 for v in per.values()) and len(per) == 16, arm
    return {k: (st.mean(a for a, _ in v), st.mean(b for _, b in v)) for k, v in per.items()}

def ninfer(arm):
    d = json.loads((O / arm / "report.json").read_text())
    out = {}
    for s in d["streams"]:
        w = s["windows"][1]
        assert w["scored_tokens"] == 1024 and w["input_begin"] == 1024, (arm, s["id"])
        out[int(s["id"][5:])] = w["mean_nll"]
    assert len(out) == 16, arm
    return out

def ci(xs):
    m = st.mean(xs); h = T95 * st.stdev(xs) / math.sqrt(len(xs))
    return f"{m:+.5f} [{m - h:+.5f}, {m + h:+.5f}]"

rows = []
for depth, halo_arms, nin_arms in [(65536, ["int8-65536", "k8v4-65536", "q4-65536"], ["rk4v4e8-65536", "int8-65536"]),
                                   (140288, ["int8-140288-ring", "k8v4-140288"], ["rk4v4e8-140288"])]:
    hs = {a: halo(a) for a in halo_arms}
    base = {c: v[1] for c, v in hs[halo_arms[0]].items()}
    for a in halo_arms:   # every Halocline arm scores the same base rows
        assert all(abs(hs[a][c][1] - base[c]) < 1e-6 for c in base), a
    print(f"\n## depth {depth:,} (16 chunks x 1,024 rows; base PPL {math.exp(st.mean(base.values())):.4f})\n")
    print("| engine, KV | PPL | dNLL vs BF16 [95% CI] | vs Halocline K8V4, paired [95% CI] |\n|---|---|---|---|")
    k8 = next(a for a in halo_arms if a.startswith("k8v4"))
    for a in halo_arms:
        d = [hs[a][c][0] - base[c] for c in sorted(base)]
        p = [hs[a][c][0] - hs[k8][c][0] for c in sorted(base)]
        print(f"| Halocline, {a.split('-')[0]} | {math.exp(st.mean(hs[a][c][0] for c in base)):.4f} | {ci(d)} | {'(reference)' if a == k8 else ci(p)} |")
    for a in nin_arms:
        if not (O / a / "report.json").exists():
            print(f"| NInfer, {a.split('-')[0]} | not run | | |"); continue
        n = ninfer(a)
        d = [n[c] - base[c] for c in sorted(base)]
        p = [n[c] - hs[k8][c][0] for c in sorted(base)]
        print(f"| NInfer, {a.split('-')[0]} | {math.exp(st.mean(n.values())):.4f} | {ci(d)} | {ci(p)} |")
