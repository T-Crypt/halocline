#!/usr/bin/env python3
"""Compare two .halo packs' section directories (name, size, sha256 as stored by the converter)."""
import sys
sys.path.insert(0, sys.argv[1])            # a Halocline release tree (for serve.engine.modelfile)
from serve.engine.modelfile import ModelFile
a, b = ModelFile(sys.argv[2]), ModelFile(sys.argv[3])
names = sorted(set(a.sections) | set(b.sections))
same = diff = 0
for n in names:
    ea, eb = a.sections.get(n), b.sections.get(n)
    if ea is None or eb is None:
        print(f"ONLY {'new' if ea is None else 'old'}  {n}  {(eb or ea)[1] if len(eb or ea) > 1 else ''}")
        continue
    ha, hb = ea[2].hex(), eb[2].hex()
    ok = ha == hb and ea[1] == eb[1]
    same += ok; diff += not ok
    print(f"{'SAME' if ok else 'DIFF'}  {n:22s} size {ea[1]} / {eb[1]}  sha {ha[:16]} / {hb[:16]}")
print(f"shared sections: {same} identical, {diff} different")
