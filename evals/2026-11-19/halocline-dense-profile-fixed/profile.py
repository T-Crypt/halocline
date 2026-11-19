#!/usr/bin/env python3
"""Dense window profile at FIXED widths on the served code (release 626f0add = aa156951 + W37 flags), so each table is
one width T and greedy text repeats: --spec T --draft-ahead T-1 --draft-min T-1 (T=1: --spec 1, no drafting, so no
--draft-vocab: the engine refuses it without the drafter).
--long-context --draft-vocab 32K --vision cpu (the served flags) plus --profile. One fresh server per (T, workload);
tables read from GET /api/logs. Not for absolute ms (uncaptured, PDL off): shares and steps.  profile.py OUTDIR"""
import json, random, signal, subprocess, sys, time, urllib.request
from pathlib import Path
B = Path.home() / "halocline/releases/626f0add"
PACK = "/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo"
PY = str(Path.home() / "halocline/releases/b22ecbb2/.venv/bin/python")
V32 = str(B / "tools/draft_vocab/data/qwen38-27b-wikitext103-src-n32768.bin")
URL = "http://127.0.0.1:8095"
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return "\n".join(" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(tokens // 18))
def get(path, body=None, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def run(out, T, depth):
    name = f"T{T}-{'short' if depth == 0 else 'd137k'}"
    flags = ["--spec", "1"] if T == 1 else ["--spec", str(T), "--draft-ahead", str(T - 1), "--draft-min", str(T - 1)]
    cmd = [str(B / "bin/halocline"), "serve", PACK, "--python", PY, "--host", "127.0.0.1", "--port", "8095", "--long-context",
           "--vision", "cpu", "--profile", *flags] + ([] if T == 1 else ["--draft-vocab", V32])
    srv = subprocess.Popen(cmd, stdout=open(out / f"{name}.serve.out", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                m = get("/v1/models", timeout=5)["data"][0]
                if m.get("status", {}).get("value", "loaded") == "loaded": break
            except Exception:
                pass
            if srv.poll() is not None: raise SystemExit(f"{name}: server exited early")
            time.sleep(1)
        content = (filler(depth) + "\n\nIgnore the lines above. " if depth else "") + PROSE
        t = time.time()
        d = get("/v1/chat/completions", {"model": m["id"], "max_tokens": 1200, "temperature": 0,
                 "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]})
        wall = time.time() - t; time.sleep(2)
        lines = get("/api/logs?limit=500")["lines"]
        prof = [l["msg"] for l in lines if "halo dense profile" in l["msg"]]
        (out / f"{name}.profile.txt").write_text("\n".join(prof) + "\n")
        tm = d.get("timings", {})
        res = {"name": name, "T": T, "completion_tokens": d["usage"]["completion_tokens"], "wall_s": round(wall, 1),
               "decode_tps": tm.get("predicted_per_second"), "profile_lines": len(prof)}
        print(json.dumps(res), flush=True); return res
    finally:
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(30)
        except subprocess.TimeoutExpired: srv.kill(); srv.wait()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
res = [run(out, T, 0) for T in (1, 2, 3, 4, 5, 6)] + [run(out, T, 136000) for T in (1, 4)]
(out / "results.json").write_text(json.dumps(res, indent=1))
