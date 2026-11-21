#!/usr/bin/env python3
"""CPU image encoder thread sweep, 2026-11-21: halocline-vision (release b2f81178, unchanged) on the CPU, --threads N,
one warm-up then 6 of the 19 airbench images (/var/tmp/vision-ab/img, every third), wall time per ENC line. Card not used."""
import subprocess, sys, time, os, statistics as st, pathlib, tempfile
R = os.path.expanduser("~/halocline/releases/b2f81178")
PACK = "/data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL-vision.halo"
IMG = sorted(pathlib.Path("/var/tmp/vision-ab/img").glob("*"))
out = tempfile.mkdtemp(prefix="vt-", dir="/var/tmp")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
for n in [int(x) for x in sys.argv[1:]] or [16, 24, 32, 8]:
    p = subprocess.Popen([f"{R}/bin/halocline-vision", "--model", PACK, "--threads", str(n)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
    while not (l := p.stdout.readline()).startswith(("READY", "ERR")) and l: pass
    ts = []
    for i, im in enumerate([IMG[0]] + IMG[::3][:6]):
        t = time.perf_counter(); p.stdin.write(f"ENC {im} {out}/{i}.bin\n"); r = p.stdout.readline()
        if not r.startswith("OK"): print("ERR", n, im, r.strip()); break
        if i: ts.append(time.perf_counter() - t)
    p.stdin.close(); p.wait()
    print(f"threads={n:2d} images={len(ts)} sum={sum(ts):6.1f}s median={st.median(ts):5.2f}s max={max(ts):5.2f}s", flush=True)
