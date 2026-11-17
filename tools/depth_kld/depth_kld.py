"""KL-divergence bases whose scored windows sit after a long prefix: the long-context quality gate (docs/roadmap.md P0).

A stock `llama-perplexity --kl-divergence-base` file scores the second half of every chunk (positions n_ctx/2 ..
n_ctx-2), so its windows never sit deeper than half the context and a window after a 200K prefix would cost a 400K
context and 100 GB of rows. With llama-kl-first.patch (beside this file) llama-perplexity takes LLAMA_KL_FIRST=D and
scores positions D .. n_ctx-2 instead; with n_ctx = D + W + 1 each chunk is a D-token prefix and a W-row window. The
file starts with "_logitsF" and stores D after n_chunk (i32); everything else is llama.cpp's layout, and both
llama.cpp's --kl-divergence and Halocline's quality gate (include/halo/program/ppl_gate.hpp, halocline-bench --quality)
read it.

The patch is against llama.cpp 2cbe0ec (git -C LLAMA apply tools/depth_kld/llama-kl-first.patch, then build
llama-perplexity with -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89). Without LLAMA_KL_FIRST the patched binary writes a
base byte-identical to stock llama.cpp's.

    depth_kld.py prepare --parquet TRAIN-00000.parquet --bytes 16000000 --out corpus.txt   # needs pyarrow
    depth_kld.py plan --model BF16-00001-of-00002.gguf --text corpus.txt --llama-bin DIR --out-dir DIR \\
        --depths 65536,140288,188416,261631 [--window 512] [--chunks 8] [--run]
    depth_kld.py check BASE.kld [BASE.kld ...]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import struct
import subprocess
import sys
from pathlib import Path

MAGIC_STOCK, MAGIC_DEPTH = b"_logits_", b"_logitsF"
MODEL_CONTEXT = 262144          # Qwen3.8-27B's native context (qwen35.context_length)
# Qwen3.8-27B: 16 attention layers (every 4th of 64), 4 K/V heads of 256: f16 K and V per token
KV_BYTES_PER_TOKEN = 16 * 2 * 4 * 256 * 2
BF16_BYTES = 49986159616 + 4671576000   # unsloth/Qwen3.8-27B-GGUF@4af51b35 BF16/*.gguf
N_VOCAB = 248320
GIB = 1 << 30
LAYER_GIB = BF16_BYTES / GIB / 64   # one of 64 layers' BF16 weights, the embedding and head spread over them (an upper bound)
CARD_GIB, COMPUTE_GIB = 23.0, 2.5   # usable on a 24 GiB card; llama.cpp's compute buffers at -ub 4096 with -fa on


def row_bytes(n_vocab: int) -> int:
    """One scored row: two floats (scale, min log-prob) and n_vocab uint16, padded to an even count (llama.cpp's nv)."""
    return 2 * (2 * ((n_vocab + 1) // 2) + 4)


def base_bytes(n_ctx: int, n_chunk: int, first: int, n_vocab: int, depth: bool) -> int:
    return (24 if depth else 20) + 4 * n_ctx * n_chunk + n_chunk * (n_ctx - 1 - first) * row_bytes(n_vocab)


def read_header(path) -> dict:
    """The header of a stock or depth base, with its expected size; raises ValueError on a file that is neither."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 20 or head[:8] not in (MAGIC_STOCK, MAGIC_DEPTH):
        raise ValueError(f"{path}: not a llama.cpp KL-divergence base")
    depth = head[:8] == MAGIC_DEPTH
    n_ctx, n_vocab, n_chunk = struct.unpack_from("<Iii", head, 8)
    if depth and len(head) < 24:
        raise ValueError(f"{path}: truncated depth header")
    first = struct.unpack_from("<i", head, 20)[0] if depth else n_ctx // 2
    if n_ctx < 4 or n_vocab <= 0 or n_chunk < 1 or not 1 <= first <= n_ctx - 2:
        raise ValueError(f"{path}: n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk {n_chunk}, first {first}")
    return {"depth": depth, "n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk, "first": first,
            "window": n_ctx - 1 - first, "bytes": base_bytes(n_ctx, n_chunk, first, n_vocab, depth)}


def check(paths) -> int:
    bad = 0
    for p in paths:
        try:
            h = read_header(p)
        except ValueError as e:
            print(f"FAIL {e}")
            bad += 1
            continue
        size = os.path.getsize(p)
        ok = size == h["bytes"]
        bad += not ok
        want = "" if ok else f" (expected {h['bytes']})"
        print(f"{'ok  ' if ok else 'FAIL'} {p}: {'depth' if h['depth'] else 'stock'} base, n_ctx {h['n_ctx']}, scored from "
              f"{h['first']} ({h['window']} rows per chunk), {h['n_chunk']} chunks, n_vocab {h['n_vocab']}, {size} bytes{want}")
    return 1 if bad else 0


class Step:
    """One depth's llama-perplexity run: its context, output file and estimates."""

    def __init__(self, depth: int, window: int, chunks: int, out_dir: Path):
        self.depth, self.window, self.chunks = depth, window, chunks
        self.n_ctx = depth + window + 1
        if depth < 1 or window < 1 or chunks < 1:
            raise ValueError("depth, window and chunks must be positive")
        if self.n_ctx > MODEL_CONTEXT:
            raise ValueError(f"depth {depth} + window {window} + 1 = {self.n_ctx} is past the model's {MODEL_CONTEXT} "
                             f"context; the deepest window is depth {MODEL_CONTEXT - window - 1}")
        self.out = out_dir / f"qwen38-27b-bf16-depth{depth}-w{window}-c{chunks}.kld"

    def bytes(self) -> int:
        return base_bytes(self.n_ctx, self.chunks, self.depth, N_VOCAB, True)

    def kv_gib(self) -> float:
        return self.n_ctx * KV_BYTES_PER_TOKEN / GIB

    def tokens(self) -> int:
        return self.n_ctx * self.chunks

    def auto_ngl(self) -> int:
        """Layers that fit on the card beside this depth's f16 K/V and the compute buffers (an estimate: lower it on OOM)."""
        return max(0, int((CARD_GIB - COMPUTE_GIB - self.kv_gib()) / LAYER_GIB))

    def host_gib(self, ngl: int) -> float:
        return BF16_BYTES / GIB - ngl * LAYER_GIB

    def argv(self, llama_bin: Path, model: Path, text: Path, batch: int, ngl: int, threads: int) -> list[str]:
        return [str(llama_bin / "llama-perplexity"), "-m", str(model), "-f", str(text), "-c", str(self.n_ctx),
                "--chunks", str(self.chunks), "-b", str(batch), "-ub", str(batch), "-fa", "on", "-ngl", str(ngl),
                "-t", str(threads), "--kl-divergence-base", str(self.out)]


def plan(a) -> int:
    out_dir = Path(a.out_dir)
    try:
        steps = [Step(int(d), a.window, a.chunks, out_dir) for d in a.depths.split(",")]
    except ValueError as e:
        print(f"plan: {e}", file=sys.stderr)
        return 2
    need = max(s.tokens() for s in steps)
    print(f"# {len(steps)} depth bases, {a.window} scored rows per chunk, {a.chunks} chunks each; the corpus must tokenize to "
          f"at least {need:,} tokens (about {need * 4.2 / 1e6:.0f} MB of English text)")
    print(f"# BF16 weights {BF16_BYTES / GIB:.1f} GiB: each run's f16 K/V and -ngl layers go on the card, the rest stays "
          f"mmap'd in host RAM (llama.cpp streams it to the card per batch)")
    total = 0
    for s in steps:
        s.ngl = s.auto_ngl() if a.ngl < 0 else a.ngl
        total += s.bytes()
        env = f"LLAMA_KL_FIRST={s.depth}"
        print(f"\n# depth {s.depth:,}: n_ctx {s.n_ctx:,}, K/V f16 {s.kv_gib():.1f} GiB + {s.ngl} layers on the card, "
              f"~{s.host_gib(s.ngl):.0f} GiB of weights in host RAM, base {s.bytes() / GIB:.2f} GiB")
        print(env + " " + shlex.join(s.argv(Path(a.llama_bin), Path(a.model), Path(a.text), a.batch, s.ngl, a.threads)))
    print(f"\n# all bases: {total / GIB:.1f} GiB in {out_dir}; check them with: {Path(__file__).name} check {out_dir}/*.kld")
    if not a.run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in steps:
        if s.out.exists() and not a.force:
            print(f"skip: {s.out} exists (--force to redo)")
            continue
        log = s.out.with_suffix(".log")
        env = dict(os.environ, LLAMA_KL_FIRST=str(s.depth))
        with open(log, "w") as f:
            f.write(f"# LLAMA_KL_FIRST={s.depth} " + shlex.join(s.argv(Path(a.llama_bin), Path(a.model), Path(a.text), a.batch,
                                                                         s.ngl, a.threads)) + "\n")
            f.flush()
            rc = subprocess.run(s.argv(Path(a.llama_bin), Path(a.model), Path(a.text), a.batch, s.ngl, a.threads),
                                env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or check([s.out]) != 0:
            print(f"FAIL depth {s.depth}: llama-perplexity exit {rc}, log {log}", file=sys.stderr)
            return 1
    return 0


def prepare(a) -> int:
    """The first `--bytes` of wikitext-103's train text (the parquet's `text` column, rows joined as stored), cut at a
    line end, with its SHA-256: a fixed corpus every arm tokenizes the same way."""
    import pyarrow.parquet as pq
    text = "".join(pq.read_table(a.parquet, columns=["text"]).column("text").to_pylist())
    data = text.encode("utf-8")[: a.bytes]
    data = data[: data.rfind(b"\n") + 1]
    Path(a.out).write_bytes(data)
    print(f"{a.out}: {len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="cut a fixed corpus from wikitext-103's train parquet")
    p.add_argument("--parquet", required=True)
    p.add_argument("--bytes", type=int, default=16_000_000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="the llama-perplexity commands per depth, with sizes; --run runs them in order")
    p.add_argument("--model", required=True, help="the BF16 GGUF (first shard)")
    p.add_argument("--text", required=True)
    p.add_argument("--llama-bin", required=True, help="the patched llama.cpp build's bin directory")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--depths", default="65536,140288,188416,261631")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--ngl", type=int, default=-1, help="layers on the card; -1 (default) sizes it per depth")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--run", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("check", help="validate base files: header and size")
    p.add_argument("bases", nargs="+")
    a = ap.parse_args(argv)
    return {"prepare": prepare, "plan": plan, "check": lambda a: check(a.bases)}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
