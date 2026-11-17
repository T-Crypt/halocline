"""tok.py GGUF enc TEXT > ids | tok.py GGUF dec IDS_FILE: the model's own tokenizer (tools/tokenizer.py)."""
import sys, re
sys.path.insert(0, "~/wt/halocline-import/tools")
from tokenizer import Tokenizer
t = Tokenizer.from_gguf(sys.argv[1])
if sys.argv[2] == "enc":
    print(",".join(map(str, t.encode(sys.argv[3], parse_special=True))))
else:
    ids = [int(x) for x in re.findall(r"\d+", open(sys.argv[3]).read())]
    print(t.decode(ids))
