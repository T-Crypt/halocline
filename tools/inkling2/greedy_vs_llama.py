#!/usr/bin/env python3
"""Greedy identity of the inkling2 forward against llama.cpp (Unsloth b12588-mix-c0c3de8): the same prompt ids, N
greedy tokens from each, and where they first differ.

    greedy_vs_llama.py --server http://127.0.0.1:PORT --text "prompt" --halo BUILD/bin/inkling2_greedy --pack PACK -n 32

The llama-server holds the same GGUF the pack was made from; /tokenize gives the ids both sides read (with the special
tokens parsed), /completion at temperature 0 the reference continuation."""
import argparse, json, subprocess, tempfile, urllib.request


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--halo", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("-n", type=int, default=32)
    a = ap.parse_args()
    ids = post(a.server + "/tokenize", {"content": a.text, "parse_special": True})["tokens"]
    ref = post(a.server + "/completion", {"prompt": ids, "n_predict": a.n, "temperature": 0, "top_k": 1,
                                          "return_tokens": True, "cache_prompt": False})["tokens"]
    with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
        f.write(" ".join(map(str, ids)))
    p = subprocess.run([a.halo, a.pack, f.name, str(a.n)], capture_output=True, text=True)
    print(p.stderr.strip())
    got = [int(x) for x in p.stdout.split()]
    first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
    print(f"prompt {len(ids)} ids; llama.cpp {ref[:16]}...\nhalocline {got[:16]}...")
    print("IDENTICAL" if first is None and len(got) == len(ref) else f"first difference at token {first}")


if __name__ == "__main__":
    main()
