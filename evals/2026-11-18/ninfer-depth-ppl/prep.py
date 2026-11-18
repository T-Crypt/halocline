#!/usr/bin/env python3
"""NInfer perplexity input on exactly the depth bases' tokens (2026-11-18, roadmap D7 "NInfer quality").

NInfer exposes no logits, so the comparison is target-token NLL on the scored rows, against the BF16 NLL of the same
rows. Each depth base (tools/depth_kld) holds 16 chunks of n_ctx = D + 1025 token ids; rows D+1 .. D+1024 are scored.
ninfer-perplexity windows a stream: window 1 = [0, C), then end = C + stride, begin = end - C. With a 1,024-token
filler in front of each chunk, C = n_ctx and stride 1,024, window 2's input is the chunk exactly and its targets are
the chunk's last 1,024 tokens. Text comes from the chunk's ids through Halocline's /detokenize (the same vocabulary);
every stream is re-tokenized through /tokenize and must give filler + chunk ids exactly, else this script stops.

    prep.py BASE.kld OUTDIR [--server http://127.0.0.1:8093]
"""
import json, struct, sys, urllib.request
from pathlib import Path

def call(server, path, body):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    server = sys.argv[sys.argv.index("--server") + 1] if "--server" in sys.argv else "http://127.0.0.1:8093"
    with open(base, "rb") as f:
        head = f.read(24)
        assert head[:8] == b"_logitsF", "not a depth base"
        n_ctx, n_vocab, n_chunk = struct.unpack_from("<Iii", head, 8)
        first = struct.unpack_from("<i", head, 20)[0]
        toks = struct.unpack(f"<{n_ctx * n_chunk}i", f.read(4 * n_ctx * n_chunk))
    assert n_ctx - 1 - first == 1024, (n_ctx, first)
    (out / "data").mkdir(parents=True, exist_ok=True)
    # the filler: plain words, ending in a paragraph break, exactly 1,024 tokens on its own
    word = " the"
    filler = "Filler" + word * 4000
    ids = call(server, "/tokenize", {"content": filler, "parse_special": False})["tokens"][:1022]
    filler = call(server, "/detokenize", {"tokens": ids})["content"] + ".\n\n"
    f_ids = call(server, "/tokenize", {"content": filler, "parse_special": False})["tokens"]
    streams, report = [], []
    for c in range(n_chunk):
        chunk = list(toks[c * n_ctx:(c + 1) * n_ctx])
        text = call(server, "/detokenize", {"tokens": chunk})["content"]
        back = call(server, "/tokenize", {"content": text, "parse_special": False})["tokens"]
        whole = call(server, "/tokenize", {"content": filler + text, "parse_special": False})["tokens"]
        ok_chunk, ok_whole = back == chunk, whole == f_ids + chunk
        report.append({"chunk": c, "filler_tokens": len(f_ids), "chunk_tokens": len(chunk),
                       "roundtrip": ok_chunk, "with_filler": ok_whole})
        if not (ok_chunk and ok_whole and len(f_ids) == 1024):
            print(json.dumps(report[-1])); sys.exit(f"chunk {c}: tokens do not round-trip; not writing a corpus")
        p = out / "data" / f"chunk{c:02d}.txt"
        p.write_text(filler + text, encoding="utf-8")
        streams.append({"id": f"chunk{c:02d}", "domain": "depth", "path": f"data/chunk{c:02d}.txt"})
    (out / "manifest.json").write_text(json.dumps({"corpus_id": f"halocline-depth{first}-w1024-c{n_chunk}", "streams": streams,
        "modes": {"quick": [x["id"] for x in streams[:1]], "full": [x["id"] for x in streams]}}, indent=2))
    (out / "prep.json").write_text(json.dumps({"base": str(base), "n_ctx": n_ctx, "first": first, "n_chunk": n_chunk,
                                               "context": n_ctx, "stride": 1024, "chunks": report}, indent=2))
    print(f"ok: {n_chunk} streams, filler {len(f_ids)} tokens, context {n_ctx}, stride 1024 -> {out}")

main()
