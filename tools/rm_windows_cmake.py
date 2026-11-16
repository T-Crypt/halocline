"""Halocline import, wave 1b: resolve the Windows conditionals in the build files as on Linux (operator: Linux for good).
if(WIN32|MSVC) ... [else() ...] endif()      -> keep the else body
if(NOT WIN32|NOT MSVC) ... [else() ...] endif() -> keep the if body
Nested ifs are tracked; elseif() in a Windows block is refused (handled by hand)."""
import re, sys
from pathlib import Path
COND = re.compile(r'^(\s*)if\(\s*(NOT\s+)?(WIN32|MSVC)\s*\)\s*$')
def resolve(lines):
    out=[]; i=0; n=0
    while i < len(lines):
        m = COND.match(lines[i])
        if not m: out.append(lines[i]); i+=1; continue
        neg = bool(m.group(2)); depth=0; body=[]; other=[]; cur=body; j=i
        while True:
            l=lines[j].strip()
            if re.match(r'if\s*\(', l): depth+=1
            if depth==1 and re.match(r'elseif\s*\(', l): raise SystemExit(f'elseif at line {j+1}: by hand')
            if depth==1 and re.match(r'else\s*\(', l): cur=other; j+=1; continue
            if re.match(r'endif\s*\(', l):
                depth-=1
                if depth==0: break
            if j!=i: cur.append(lines[j])
            j+=1
        keep = body if neg else other
        ind = len(m.group(1))
        out += resolve([k[2:] if k.startswith(' '*(ind+2)) and ind==0 and False else k for k in keep])
        i=j+1; n+=1
    resolve.count = getattr(resolve,'count',0)+n
    return out
for p in sys.argv[1:]:
    resolve.count=0
    t=Path(p).read_text().split('\n'); Path(p).write_text('\n'.join(resolve(t)))
    print(p, 'resolved', resolve.count)
