#!/usr/bin/env python3
"""W13 dense profiler, one fresh server per workload so each table is one depth and one output kind.

The engine's stderr (where `halo dense profile` prints, every 200 windows and when a session ends) goes to the
server's log buffer, not the terminal: each workload's lines are read from GET /api/logs before the server stops.
Workloads: prose and structured output (JSON) after a short prompt, after ~36K and after ~137K tokens of the smoke's
needle filler (lsw_smoke.needle's text, seed 1). Decode runs to ~1,200 tokens so a window table prints at least once.
--profile runs the window uncaptured with PDL off: compare stage shares, not absolute ms.

    profile.py OUTDIR      (run under: berth train --as halocline-dense-profile -- run.sh)
"""
import json, os, random, signal, subprocess, sys, time, urllib.request
from pathlib import Path

W = Path.home() / "wt/halocline-dense-profile"
PACK = "/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo"
PY = str(Path.home() / "halocline/releases/b22ecbb2/.venv/bin/python")
URL = "http://127.0.0.1:8095"
CMD = [str(W / "bin/halocline"), "serve", PACK, "--python", PY, "--host", "127.0.0.1", "--port", "8095", "--spec", "8",
       "--max-context", "175616", "--kv", "k8v4", "--mtp-kv", "int8", "--vision", "cpu", "--profile"]
PROSE = "Write a long story, about 900 words, about a lighthouse keeper who finds a radio that plays tomorrow's news."
STRUCT = ("Output only a JSON array of 40 objects, one per fictional employee, each with the fields id (integer), name, "
          "city, department, salary (integer), start_date (YYYY-MM-DD) and skills (an array of three strings).")

def filler(tokens):
    random.seed(1)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    return "\n".join(" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(tokens // 18))

def get(path, body=None, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def logs():   # the newest 500 lines (LOG_MAX_LINES); one workload's run is well under that
    return get("/api/logs?limit=500")["lines"]

def run(out, name, depth, task):
    srv = subprocess.Popen(CMD, stdout=open(out / f"{name}.serve.out", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                model = get("/v1/models", timeout=5)["data"][0]["id"]; break
            except Exception:
                if srv.poll() is not None: raise SystemExit(f"{name}: server exited early")
                time.sleep(1)
        content = (filler(depth) + "\n\nIgnore the lines above. " if depth else "") + task
        t = time.time()
        d = get("/v1/chat/completions", {"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": content}]})
        wall = time.time() - t
        time.sleep(2)
        lines = logs()
        (out / f"{name}.logs.json").write_text(json.dumps(lines, indent=0))
        prof = [l["msg"] for l in lines if "halo dense profile" in l["msg"]]
        (out / f"{name}.profile.txt").write_text("\n".join(prof) + "\n")
        tm = d.get("timings", {})
        res = {"name": name, "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
               "wall_s": round(wall, 1), "prompt_tps": tm.get("prompt_per_second"), "decode_tps": tm.get("predicted_per_second"),
               "profile_lines": len(prof), "log_lines": len(lines)}
        print(json.dumps(res), flush=True)
        return res
    finally:
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(30)
        except subprocess.TimeoutExpired: srv.kill(); srv.wait()

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    res = [run(out, n, dep, task) for n, dep, task in [
        ("short-prose", 0, PROSE), ("short-struct", 0, STRUCT),
        ("d36k-prose", 36000, PROSE), ("d36k-struct", 36000, STRUCT),
        ("d137k-prose", 136000, PROSE), ("d137k-struct", 136000, STRUCT)]]
    (out / "results.json").write_text(json.dumps(res, indent=1))

main()
