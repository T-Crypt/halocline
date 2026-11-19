#!/usr/bin/env python3
"""Weight bytes a decode window reads per profile stage, from the GGUF the pack copies (quantized tensors as published).
stage_bytes.py GGUF HALOCLINE_TREE"""
import sys, pathlib, collections, re
sys.path.insert(0, sys.argv[2] + "/tools")
from gguf_reader import GGUFFile
g = GGUFFile(pathlib.Path(sys.argv[1]))
S = collections.Counter(); other = collections.Counter()
rules = [  # (regex on tensor name, stage)
 (r"blk\.\d+\.ffn_(gate|up)\.weight", "FfnUp"), (r"blk\.\d+\.ffn_down\.weight", "FfnDown"),
 (r"blk\.\d+\.attn_(q|k|v)\.weight", "AttnPre"), (r"blk\.\d+\.attn_output\.weight", "AttnOut"),
 (r"blk\.\d+\.(attn_qkv|attn_gate)\.weight", "GdnPre"), (r"blk\.\d+\.ssm_out\.weight", "GdnOut"),
 (r"^output\.weight$", "Head"), (r"^token_embd\.weight$", "Embed(row only)"),
 (r"mtp|nextn", "MTP"),
]
for t in g.tensors:
    n = t.name; b = t.expected_bytes() or 0
    for rx, st in rules:
        if re.search(rx, n): S[st] += b; break
    else:
        other[n.split('.')[-2] if '.' in n else n] += b
tot = sum(v for k, v in S.items() if not k.startswith("Embed")) + sum(other.values())
print(f"total read per window (excl. embedding table): {tot/2**30:.2f} GiB")
for k, v in sorted(S.items(), key=lambda x: -x[1]): print(f"{k:16s} {v/2**20:9.1f} MiB  {100*v/tot:5.1f}%")
print("unmatched:", {k: round(v/2**20, 1) for k, v in other.most_common(12)})
