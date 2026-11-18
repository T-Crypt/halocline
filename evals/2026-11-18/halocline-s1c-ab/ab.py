#!/usr/bin/env python3
"""Dense decode A/B: arms are build dirs (each with bin/halocline), served flags, one fresh server per (arm, workload),
1,200 decoded tokens a request so the decode rate is a long average. Workloads: prose and structured output after a
short prompt, ~36K and ~137K tokens of the smoke's needle filler (seed 1). The order of arms is the argument order
(write it ABBA). Greedy (temperature 0): a bitwise kernel change gives the same tokens and the same
acceptance, so the decode rate isolates the kernel. (Served sampling, temperature 1.0, moved short prose 79.9 to 156.1
t/s between two runs of one build: noise-sampled/.)

    ab.py OUTDIR LABEL=BUILD_DIR[::EXTRA FLAGS] [...]      (run under berth train)
"""
import hashlib, json, random, signal, subprocess, sys, time, urllib.request
from pathlib import Path

PACK = "/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo"
PY = str(Path.home() / "halocline/releases/b22ecbb2/.venv/bin/python")
URL = "http://127.0.0.1:8095"
FLAGS = ["--host", "127.0.0.1", "--port", "8095", "--spec", "8", "--max-context", "175616", "--kv", "k8v4",
         "--mtp-kv", "int8", "--vision", "cpu"]
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
STRUCT = ("Output only a JSON array of 40 objects, one per fictional employee, each with the fields id (integer), name, "
          "city, department, salary (integer), start_date (YYYY-MM-DD) and skills (an array of three strings).")
WORK = [("short-prose", 0, PROSE), ("short-struct", 0, STRUCT), ("d36k-prose", 36000, PROSE),
        ("d36k-struct", 36000, STRUCT), ("d137k-prose", 136000, PROSE), ("d137k-struct", 136000, STRUCT)]

def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return "\n".join(" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(tokens // 18))

def get(path, body=None, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def one(out, tag, build, name, depth, task, extra=()):
    srv = subprocess.Popen([str(Path(build) / "bin/halocline"), "serve", PACK, "--python", PY] + FLAGS + list(extra),
                           stdout=open(out / f"{tag}-{name}.serve.out", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                model = get("/v1/models", timeout=5)["data"][0]["id"]; break
            except Exception:
                if srv.poll() is not None: raise SystemExit(f"{tag} {name}: server exited early")
                time.sleep(1)
        content = (filler(depth) + "\n\nIgnore the lines above. " if depth else "") + task
        d = get("/v1/chat/completions", {"model": model, "max_tokens": 1200, "temperature": 0, "messages": [{"role": "user", "content": content}]})
        tm = d.get("timings", {})
        r = {"arm": tag, "work": name, "prompt_tokens": d["usage"]["prompt_tokens"],
             "completion_tokens": d["usage"]["completion_tokens"], "decode_tps": tm.get("predicted_per_second"),
             "prompt_tps_srv": tm.get("prompt_per_second"),
             "text_sha": hashlib.sha256(json.dumps(d["choices"][0]["message"], sort_keys=True).encode()).hexdigest()[:16]}
        print(json.dumps(r), flush=True)
        return r
    finally:
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(30)
        except subprocess.TimeoutExpired: srv.kill(); srv.wait()

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    res = []
    for i, arg in enumerate(sys.argv[2:], 1):
        label, build = arg.split("=", 1); build, _, ex = build.partition("::")
        for w in WORK: res.append(one(out, f"{i}-{label}", build, *w, extra=ex.split()))
    (out / "results.json").write_text(json.dumps(res, indent=1))
    by = {}
    for r in res: by.setdefault((r["arm"].split("-", 1)[1], r["work"]), []).append(r["decode_tps"])
    labels = sorted({k[0] for k in by})
    print("| workload | " + " | ".join(labels) + " |\n|---" * 1 + "|---" * len(labels) + "|")
    for w, *_ in WORK:
        print(f"| {w} | " + " | ".join(", ".join(f"{x:.1f}" for x in by.get((l, w), [])) for l in labels) + " |")

main()
