#!/usr/bin/env python3
"""Quant type mix (MiB) per profile stage in the GGUF.  stage_types.py GGUF HALOCLINE_TREE"""
import sys, pathlib, collections, re
sys.path.insert(0, sys.argv[2] + "/tools")
from gguf_reader import GGUFFile
g = GGUFFile(pathlib.Path(sys.argv[1]))
rules = [(r"blk\.\d+\.ffn_(gate|up)\.weight", "ffn-up"), (r"blk\.\d+\.ffn_down\.weight", "ffn-down"),
 (r"blk\.\d+\.attn_(q|k|v)\.weight", "attn-pre"), (r"blk\.\d+\.attn_output\.weight", "attn-out"),
 (r"blk\.\d+\.(attn_qkv|attn_gate)\.weight", "gdn-pre"), (r"blk\.\d+\.ssm_out\.weight", "gdn-out"), (r"^output\.weight$", "head")]
S = collections.defaultdict(collections.Counter); shapes = collections.defaultdict(set)
for t in g.tensors:
    for rx, st in rules:
        if re.search(rx, t.name):
            S[st][str(getattr(t, 'type_name', None) or getattr(t, 'ggml_type', None) or t.type)] += (t.expected_bytes() or 0) / 2**20
            shapes[st].add(tuple(t.shape) if hasattr(t, 'shape') else tuple(getattr(t, 'dims', ())))
            break
for st, c in S.items(): print(f"{st:9s}", {k: round(v) for k, v in c.most_common()}, sorted(shapes[st])[:4])
