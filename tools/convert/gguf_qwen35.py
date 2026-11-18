"""A qwen35 pack (Qwen3.5-family dense hybrid: GDN and gated attention layers, dense SwiGLU) from a llama.cpp GGUF, for
tools/convert/convert.py --gguf. Lossless: every tensor's bytes are copied as the GGUF stores them, in its type
(Unsloth Dynamic mixes IQ2_XS to Q8_0 per tensor), with no dequantization and no requantization.

What changes is names and metadata only:

  gguf      every tensor of the GGUF, bytes as stored, 32-byte aligned. The drafter layer llama.cpp stores as the last
            block (`blk.<n>.*` and `blk.<n>.nextn.*` with `qwen35.nextn_predict_layers` = 1) gets the `mtp.*` names the
            qwen35 adapter reads (MTP_NAMES below).
  meta      `gguf_kv` is the GGUF's metadata with `qwen35.block_count` lowered by the drafter layers (the adapter counts
            main layers); `tensors` gives each tensor's type, shape, bytes, offset and the SHA-256 of its source bytes.
  tokenizer/*  from the GGUF (tools/tokenizer.py).
  vision/mmproj, vision/text_vocab  with --mmproj: the projector GGUF as it is, and the text model's metadata alone.

llama.cpp's qwen35 conversion has already put the tensors in the order the adapter reads: norms with the "+1" folded
in, ssm_a = -exp(A_log), GDN value heads tiled (value head h meets key head h % group_count), conv kernels as
[kernel, channels]. checkpoint_qwen35.py writes the same conventions from an NVFP4 checkpoint.

`verify(pack, gguf, mmproj)` re-reads both files and compares every tensor's SHA-256, every `tokenizer/` section with
what the converter writes from the GGUF, the vision sections, and the metadata the converter copies: the D2 gate
(docs/roadmap.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]   # run as a script under -I: its own directory is not on the path
import model_sections as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from model_file import TENSOR_ALIGN, ModelFile, ModelWriter  # noqa: E402

ARCH = "qwen35"
A = ARCH + "."
LICENSE = {"apache-2.0": "Apache License 2.0"}
HASH_CHUNK = 64 << 20

# drafter tensor (after `blk.<n>.`) -> pack name
MTP_NAMES = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
    "attn_norm.weight": "mtp.attn_norm.weight",
    "post_attention_norm.weight": "mtp.post_attention_norm.weight",
    "attn_q_norm.weight": "mtp.attn_q_norm.weight",
    "attn_k_norm.weight": "mtp.attn_k_norm.weight",
    **{f"{r}.weight": f"mtp.{r}.weight" for r in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")},
}
# tensors of a main layer: GDN (recurrent) and attention mixers, then the FFN both have
GDN = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_a",
       "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight"}
ATTN = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"}
COMMON = {"attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
GLOBAL = {"token_embd.weight", "output_norm.weight", "output.weight"}


class Plan:
    """The pack's tensors in source order: `rows` = [(pack name, source path, absolute offset, bytes, TensorInfo)]."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        g = GGUFFile(path)
        self.kv = {k: _json_safe(v) for k, v in g.metadata.items()}
        if self.kv.get("general.architecture") != ARCH:
            raise S.ConvertError(f"{path.name}: architecture is {self.kv.get('general.architecture')!r}, not {ARCH!r}")
        blocks = int(self.kv.get(A + "block_count") or 0)
        nextn = int(self.kv.get(A + "nextn_predict_layers") or 0)
        if nextn not in (0, 1):
            raise S.ConvertError(f"{path.name}: {nextn} drafter layers; the qwen35 adapter reads one")
        self.n_layers = blocks - nextn
        self.nextn = nextn
        interval = int(self.kv.get(A + "full_attention_interval") or 0)
        if self.n_layers <= 0 or interval <= 0:
            raise S.ConvertError(f"{path.name}: block_count {blocks}, nextn_predict_layers {nextn}, "
                                 f"full_attention_interval {interval}")
        self.kv[A + "block_count"] = self.n_layers
        rec = self.kv.get(A + "attention.recurrent_layers")
        if isinstance(rec, list) and len(rec) == blocks:   # a per-layer list counts the drafter layer too
            self.kv[A + "attention.recurrent_layers"] = rec[:self.n_layers]
        self.rows, problems = [], []
        seen = set()
        for t in g.tensors:
            name = self.pack_name(t.name, nextn)
            if name is None:
                problems.append(f"{t.name}: not a tensor of a qwen35 model")
                continue
            size = t.expected_bytes()
            if size is None:
                problems.append(f"{t.name}: {t.type_name} {list(t.shape)} is not whole blocks")
                continue
            seen.add(name)
            self.rows.append((name, path, g.data_start + t.offset, size, t))
        for il in range(self.n_layers):
            recurrent = rec[il] != 0 if isinstance(rec, list) and len(rec) == blocks else (il + 1) % interval != 0
            for base in sorted(COMMON | (GDN if recurrent else ATTN)):
                if f"blk.{il}.{base}" not in seen:
                    problems.append(f"blk.{il}.{base}: missing")
        for name in sorted(GLOBAL | (set(MTP_NAMES.values()) if nextn else set())):
            if name not in seen:
                problems.append(f"{name}: missing")
        if problems:
            more = f"\n... and {len(problems) - 8} more" if len(problems) > 8 else ""
            raise S.ConvertError("\n".join(problems[:8]) + more)
        self.rows.sort(key=lambda r: r[2])   # one sequential read of the source

    def pack_name(self, name: str, nextn: int):
        if name in GLOBAL:
            return name
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if not m:
            return None
        il, base = int(m.group(1)), m.group(2)
        if il < self.n_layers:
            return name if base in GDN | ATTN | COMMON else None
        if nextn and il == self.n_layers:
            return MTP_NAMES.get(base)
        return None


def _json_safe(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    return v


def sha256_range(path, off: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = size
        while left > 0:
            b = f.read(min(HASH_CHUNK, left))
            if not b:
                raise IOError(f"{path}: short read at {off + size - left}")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def build(a, Job, fingerprint, die, export_tokenizer) -> None:
    """`export_tokenizer(gguf, dir)` writes the tokenizer files under dir/tokenizer and returns them (convert.py's)."""
    src = pathlib.Path(a.gguf[0])
    if len(a.gguf) != 1:
        die("a qwen35 GGUF is one file; merge split GGUFs first (llama-gguf-split --merge)")
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    mmproj = pathlib.Path(a.mmproj) if a.mmproj and not a.no_vision else None
    if mmproj is not None and not mmproj.is_file():
        die(f"missing input: {mmproj}")
    try:
        plan = Plan(src)
    except S.ConvertError as e:
        die(str(e))
    stem = src.stem
    quant = "UD-" + stem.split("-UD-")[-1] if "-UD-" in stem else stem.rsplit("-", 1)[-1]
    name = a.name or f"{plan.kv.get('general.basename') or plan.kv.get('general.name') or ARCH} {quant}"
    print(f"{src.name}: {plan.n_layers} layers, {len(plan.rows)} tensors, "
          f"{sum(r[3] for r in plan.rows) / (1 << 30):.3f} GiB, copied as stored", flush=True)
    if getattr(a, "dry_run", False):
        for n, _, off, size, t in plan.rows[:6] + plan.rows[-6:]:
            print(f"  {n:<40} {t.type_name:<8} {size:>12}  (source offset {off})")
        print(f"dry run: {out} not written")
        return
    print("hashing the source tensors ...", flush=True)
    t0 = time.time()
    digests = {n: sha256_range(p, off, size) for n, p, off, size, _ in plan.rows}
    print(f"  {len(digests)} tensors in {time.time() - t0:.1f} s", flush=True)
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            print("writing sections:", flush=True)
            for p in export_tokenizer(src, pathlib.Path(tmp)):
                job.file(f"tokenizer/{p.name}", p)
            t = time.time()
            offs = w.add_parts("gguf", [(str(p), off, size) for _, p, off, size, _ in plan.rows], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            if mmproj is not None:
                import pack_vision
                job.file("vision/mmproj", mmproj)
                data = pack_vision.vocab_only_gguf(str(src))
                if data is None:
                    die(f"{src.name} carries no tokenizer metadata for the vision sidecar")
                job.bytes("vision/text_vocab", data)
            tens = {n: {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.shape],
                        "bytes": size, "offset": off_, "source_sha256": digests[n]}
                    for (n, _, _, size, t_), off_ in zip(plan.rows, offs)}
            lic = str(plan.kv.get("general.license", ""))
            renamed = {f"blk.{plan.n_layers}.{k}": v for k, v in MTP_NAMES.items()} if plan.nextn else {}
            meta = {
                "format": "halo", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py", "path": "gguf_qwen35"},
                "sources": [{"path": src.name, "size": src.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src)}] +
                           ([{"path": mmproj.name, "size": mmproj.stat().st_size,
                              "fingerprint_sha256_first_last_64mib_and_size": fingerprint(mmproj)}] if mmproj else []),
                "gguf_kv": plan.kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic), **({"quantized_by": plan.kv["general.quantized_by"]}
                                                              if plan.kv.get("general.quantized_by") else {})},
                "gguf": {"lossless": "every tensor's bytes as the source GGUF stores them; source_sha256 per tensor",
                         "block_count": f"{plan.n_layers} main layers (the GGUF counts the drafter layer too)",
                         "renamed": renamed},
                "mtp": {"layers": 1, "source": "the GGUF's nextn layer, as stored"} if renamed else None,
                "vision": {"mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size} if mmproj else None,
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")


def _export_tokenizer(gguf: pathlib.Path, out_dir: pathlib.Path):
    """What the converter writes under tokenizer/, from convert.py (imported here: convert imports this module)."""
    from convert import export_tokenizer
    return export_tokenizer(gguf, out_dir)


def _verify_tensors(mf, meta: dict, plan: Plan, pack: pathlib.Path, gguf: pathlib.Path) -> list[str]:
    """Every tensor of the pack against the GGUF it names: presence, type, shape, bytes, and the SHA-256 of the source
    bytes, of the pack bytes and of the meta's record."""
    base, _ = mf.section("gguf")
    want = {n: (p, off, size, t) for n, p, off, size, t in plan.rows}
    got = meta["tensors"]
    bad = [f"{n}: in the GGUF, not in the pack" for n in sorted(set(want) - set(got))]
    bad += [f"{n}: in the pack, not in the GGUF" for n in sorted(set(got) - set(want))]
    for n in sorted(set(want) & set(got)):
        p, off, size, t = want[n]
        e = got[n]
        if (e["type_id"], e["shape"], e["bytes"]) != (t.type_id, [int(d) for d in t.shape], size):
            bad.append(f"{n}: type/shape/bytes {e['type_id']}/{e['shape']}/{e['bytes']} vs the GGUF's "
                       f"{t.type_id}/{[int(d) for d in t.shape]}/{size}")
            continue
        src_h = sha256_range(p, off, size)
        pack_h = sha256_range(pack, base + e["offset"], e["bytes"])
        if not (src_h == pack_h == e.get("source_sha256")):
            bad.append(f"{n}: SHA-256 source {src_h[:16]} pack {pack_h[:16]} meta {str(e.get('source_sha256'))[:16]}")
    return bad


def _verify_meta(meta: dict, plan: Plan, gguf: pathlib.Path) -> list[str]:
    """`gguf_kv` and `sources[0]`: the metadata the converter copies from the GGUF's header, and where it came from."""
    bad = []
    kv = meta.get("gguf_kv") or {}
    if kv != plan.kv:
        diff = sorted(k for k in set(kv) | set(plan.kv) if kv.get(k) != plan.kv.get(k))
        bad.append(f"meta gguf_kv: {len(diff)} keys differ from the GGUF's metadata ({', '.join(diff[:6])}"
                   f"{', ...' if len(diff) > 6 else ''})")
    sources = meta.get("sources") or [{}]
    src = sources[0]
    if (src.get("path"), src.get("size")) != (gguf.name, gguf.stat().st_size):
        bad.append(f"meta sources[0]: {src.get('path')!r} {src.get('size')}, the GGUF is {gguf.name!r} "
                   f"{gguf.stat().st_size:,} bytes")
    return bad


def _verify_tokenizer(mf, gguf: pathlib.Path, export) -> list[str]:
    """Every `tokenizer/*` section against the file the converter writes from the GGUF's metadata: the four tokenizer
    files, and chat_template.jinja when the GGUF carries tokenizer.chat_template."""
    have = {n: mf.read(n) for n in mf.names() if n.startswith("tokenizer/")}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            files = export(gguf, pathlib.Path(tmp))
        except ValueError as e:
            return [f"tokenizer/: the GGUF's tokenizer cannot be exported ({e})"]
        want = {f"tokenizer/{p.name}": p.read_bytes() for p in files}
    bad = [f"{n}: the converter writes it, the pack has no such section" for n in sorted(set(want) - set(have))]
    bad += [f"{n}: in the pack, not what the converter writes from the GGUF (extra file)"
            for n in sorted(set(have) - set(want))]
    for n in sorted(set(want) & set(have)):
        if want[n] != have[n]:
            bad.append(f"{n}: {len(have[n]):,} bytes in the pack, {len(want[n]):,} from the GGUF "
                       f"(SHA-256 {hashlib.sha256(have[n]).hexdigest()[:16]} vs "
                       f"{hashlib.sha256(want[n]).hexdigest()[:16]})")
    return bad


def _verify_vision(mf, meta: dict, pack: pathlib.Path, gguf: pathlib.Path, mmproj) -> list[str]:
    """`vision/mmproj` against the projector file when one is given, byte for byte; `vision/text_vocab` against the
    vocab-only GGUF the converter makes from the source; and meta.vision against the sections that are there."""
    bad = []
    names = mf.names()
    vis = meta.get("vision")
    has = "vision/mmproj" in names
    if has and not isinstance(vis, dict):
        bad.append("meta vision: the pack has vision/mmproj, meta has no vision record")
    if vis and not has:
        bad.append("vision/mmproj: meta names it, the pack has no such section")
    if has:
        off, size = mf.section("vision/mmproj")
        if isinstance(vis, dict):
            if vis.get("mmproj_bytes") != size:
                bad.append(f"meta vision.mmproj_bytes {vis.get('mmproj_bytes')}, the section is {size:,} bytes")
            if vis.get("mmproj") != "vision/mmproj":
                bad.append(f"meta vision.mmproj {vis.get('mmproj')!r}, the section is 'vision/mmproj'")
        if mmproj is not None:
            src_n = os.path.getsize(mmproj)
            src_h = sha256_range(mmproj, 0, src_n)
            pack_h = sha256_range(pack, off, size)
            if not (src_h == pack_h and src_n == size):
                bad.append(f"vision/mmproj: {size:,} bytes in the pack, {src_n:,} in {mmproj.name} "
                           f"(SHA-256 {pack_h[:16]} vs {src_h[:16]})")
    elif mmproj is not None:
        bad.append(f"vision/mmproj: --mmproj {mmproj.name} was given, the pack has no such section")
    if "vision/text_vocab" in names:
        import pack_vision
        want = pack_vision.vocab_only_gguf(str(gguf))
        got = mf.read("vision/text_vocab")
        if want is None:
            bad.append("vision/text_vocab: the GGUF carries no tokenizer metadata for it")
        elif got != want:
            bad.append(f"vision/text_vocab: {len(got):,} bytes in the pack, {len(want):,} from the GGUF's metadata")
    return bad


def verify(pack: pathlib.Path, gguf: pathlib.Path, mmproj=None, tokenizer_files=None) -> list[str]:
    """Everything of `pack` this GGUF and projector would produce, against what they are now:

      tensors        every tensor's type, shape, bytes and SHA-256 against the GGUF's, and the meta's record of each
      meta           gguf_kv against the GGUF's metadata, sources[0] against the GGUF's name and size
      tokenizer/     each section byte for byte against what convert.py's export writes from the GGUF
      vision/        mmproj byte for byte against `mmproj` when given, text_vocab against the vocab-only GGUF, and
                     meta.vision against the two sections

    Returns the mismatches; empty when the pack is what these inputs produce. `tokenizer_files` replaces the converter's
    export (tests use it); `mmproj` may be None, and then vision/mmproj is not compared."""
    mf = ModelFile(str(pack))
    meta = json.loads(mf.read("meta"))
    plan = Plan(gguf)
    bad = _verify_tensors(mf, meta, plan, pack, gguf)
    bad += _verify_meta(meta, plan, gguf)
    bad += _verify_tokenizer(mf, gguf, tokenizer_files or _export_tokenizer)
    bad += _verify_vision(mf, meta, pack, gguf, mmproj)
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check a qwen35 pack against the GGUF (and projector) it was converted "
                                             "from, section by section.")
    ap.add_argument("cmd", choices=("verify",))
    ap.add_argument("pack")
    ap.add_argument("gguf")
    ap.add_argument("--mmproj", metavar="GGUF", help="the projector GGUF the pack was built with; its bytes are "
                    "compared with vision/mmproj")
    a = ap.parse_args()
    pack, gguf = pathlib.Path(a.pack), pathlib.Path(a.gguf)
    bad = verify(pack, gguf, pathlib.Path(a.mmproj) if a.mmproj else None)
    for b in bad[:20]:
        print("MISMATCH", b)
    if len(bad) > 20:
        print(f"... and {len(bad) - 20} more mismatches")
    mf = ModelFile(str(pack))
    meta = json.loads(mf.read("meta"))
    n_tok = sum(1 for n in mf.names() if n.startswith("tokenizer/"))
    vis = [n for n in mf.names() if n.startswith("vision/")]
    if "vision/mmproj" in vis and not a.mmproj:
        print("note: vision/mmproj is not compared: pass --mmproj to check it")
    print(f"verify: {len(meta['tensors'])} tensors, {n_tok} tokenizer files, {len(vis)} vision sections, "
          f"{len(bad)} mismatches -> {'LOSSLESS' if not bad else 'FAIL'}")
    sys.exit(1 if bad else 0)
