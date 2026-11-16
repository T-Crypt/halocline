"""Halocline import, wave 1a: remove DeepSeek-V4.1-Flash (operator, 2026-11-16: "I do not want anything for Deepseek")."""
import re, subprocess, sys
from pathlib import Path
R = Path(sys.argv[1])
def cmake_strip(path, var):
    lines = (R/path).read_text().split('\n'); out=[]; i=0; cut=0
    while i < len(lines):
        if re.match(r'\s*if\(\s*%s\s*\)\s*$' % var, lines[i]):
            depth=0
            while i < len(lines):
                l=lines[i].strip()
                if re.match(r'if\s*\(', l): depth+=1
                elif re.match(r'endif\s*\(', l): depth-=1
                i+=1
                if depth==0: break
            cut+=1; continue
        out.append(lines[i]); i+=1
    (R/path).write_text('\n'.join(out)); return cut
files = subprocess.run(['git','-C',str(R),'ls-files'],capture_output=True,text=True).stdout.split('\n')
gone = [f for f in files if f and re.search(r'deepseek41|ds41|dspark', f, re.I)]
for f in gone: (R/f).unlink()
print('deleted files:', len(gone))
print('CMakeLists blocks cut:', cmake_strip('CMakeLists.txt','BASALT_MODEL_DEEPSEEK41'))
m=(R/'cmake/models.cmake').read_text(); (R/'cmake/models.cmake').write_text(m.replace('qwen4exp deepseek41 qwen35','qwen4exp qwen35'))
s=(R/'serve/models/__init__.py').read_text()
s=re.sub(r'\n    if arch == "deepseek41":\n        from serve\.models\.deepseek41 import DeepSeek41\n        return DeepSeek41\(\)','',s)
(R/'serve/models/__init__.py').write_text(s)
