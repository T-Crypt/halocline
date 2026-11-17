#!/usr/bin/env python3
"""Paired KLD comparison of KV sweep rows.tsv files.

Usage: kv_paired.py SWEEPDIR REF KV [KV ...]
Columns of rows.tsv (basalt-bench summarize_rows): 0 chunk, 1 position, 3 nll, 4 nll_base, 5 kld, 6 top1 same, 9 dlogp.
Per KV: mean KLD, PPL, top-1; against REF: mean over chunks of the per-chunk KLD difference with a 95% t interval.
"""
import math
import sys
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        c = line.split("\t")
        if len(c) < 10:
            continue
        rows[(int(c[0]), int(c[1]))] = (float(c[3]), float(c[4]), float(c[5]), int(c[6]))
    return rows


def summary(r):
    n = len(r)
    nll = sum(v[0] for v in r.values()) / n
    nllb = sum(v[1] for v in r.values()) / n
    kld = sum(v[2] for v in r.values()) / n
    top1 = 100.0 * sum(v[3] for v in r.values()) / n
    return n, math.exp(nll), math.exp(nllb), kld, top1


T975 = {9: 2.262, 19: 2.093, 29: 2.045, 49: 2.010, 99: 1.984}


def paired(a, b):
    keys = sorted(set(a) & set(b))
    per = {}
    for k in keys:
        per.setdefault(k[0], []).append(b[k][2] - a[k][2])
    d = [sum(v) / len(v) for v in per.values()]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
    df = len(d) - 1
    t = T975.get(df, 1.96)
    h = t * sd / math.sqrt(len(d)) if d else 0.0
    top = 100.0 * (sum(b[k][3] for k in keys) - sum(a[k][3] for k in keys)) / len(keys)
    return len(keys), len(d), m, m - h, m + h, top


def main():
    sweep, ref, kvs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    data = {}
    for kv in [ref] + [k for k in kvs if k != ref]:
        p = next((sweep / kv).rglob("rows.tsv"), None)
        if p is None:
            print(f"{kv}: no rows.tsv")
            continue
        data[kv] = load(p)
    print("| KV | rows | PPL | base PPL | mean KLD | top-1 %% | paired dKLD vs %s [95%% CI] | top-1 points |" % ref)
    print("|---|---|---|---|---|---|---|---|")
    for kv, r in data.items():
        n, ppl, pplb, kld, top1 = summary(r)
        if kv == ref:
            cmp_ = "(reference)"
            tp = ""
        else:
            _, nc, m, lo, hi, tp = paired(data[ref], r)
            cmp_ = f"{m:+.5f} [{lo:+.5f}, {hi:+.5f}] ({nc} chunks)"
            tp = f"{tp:+.2f}"
        print(f"| {kv} | {n} | {ppl:.4f} | {pplb:.4f} | {kld:.5f} | {top1:.2f} | {cmp_} | {tp} |")


if __name__ == "__main__":
    main()
