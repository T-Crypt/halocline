"""An inkling2 pack (Inkling 2 Mini: a looped transformer with a recurrent layer core) from a llama.cpp GGUF, for
tools/convert/convert.py --gguf. Lossless: every tensor's bytes are copied as the GGUF stores them, in its type
(Unsloth Dynamic mixes IQ2_XS to Q8_0 per tensor), with no dequantization and no requantization.

Only names and metadata change:

  gguf       every tensor of the GGUF, bytes as stored, 32-byte aligned, its GGUF name unchanged
  meta       `gguf_kv` is the GGUF's metadata as is; `tensors` gives each tensor's type, shape, bytes, offset and the
             SHA-256 of its source bytes
  tokenizer/*  from the GGUF (tools/tokenizer.py)
  vision/mmproj, vision/text_vocab  with --mmproj: the projector GGUF as it is, and the text model's metadata alone
  dspark/gguf  with --dspark: the DSpark drafter GGUF whole, its size and SHA-256 in meta.dspark

The GGUF's `inkling2.block_count` counts layer visits (each physical layer runs twice through the loop), not physical
layers: the physical layers are the `blk.0..N-1` tensors, and the loop runs `inkling2.loop.core_start..core_end`.

`verify(pack, gguf, mmproj, dspark)` re-reads both files and compares every tensor's SHA-256, every `tokenizer/`
section with what the converter writes from the GGUF, the vision and dspark sections, and the metadata the converter
copies.
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

ARCH = "inkling2"
A = ARCH + "."
LICENSE = {"apache-2.0": "Apache License 2.0"}
HASH_CHUNK = 64 << 20

# every tensor the converter accepts, with its GGUF name unchanged
GLOBAL = {
    "token_embd.weight", "token_embd_norm.weight", "output.weight", "output_norm.weight",
    "output_res_query.weight", "loop_res_query.weight", "loop_kv_res_query.weight",
}
# per physical layer, after `blk.N.`
LAYER = {
    "attn_gate.weight", "attn_k.weight", "attn_v.weight", "attn_q.weight", "attn_output.weight",
    "attn_k_conv.weight", "attn_v_conv.weight", "attn_post_conv.weight", "attn_norm.weight",
    "attn_q_clip.weight", "attn_rel.weight", "attn_rel_bias.weight", "attn_rel_clip.weight",
    "attn_res_query.weight", "attn_sink_proj.weight", "attn_k_conv_proj.weight", "attn_v_conv_proj.weight",
    "attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
    "ffn_norm.weight", "ffn_gate_inp.weight", "ffn_post_conv.weight", "ffn_res_query.weight",
    "ffn_gate_up_exps.weight", "ffn_down_exps.weight",
    "exp_probs_b.bias", "ffn_gate_up_exps.scale", "ffn_down_exps.scale",
    "loop_attn_down.weight", "loop_attn_gain.weight", "loop_attn_forget.weight",
    "loop_k_down.weight", "loop_k_gain.weight", "loop_k_forget.weight",
    "loop_v_down.weight", "loop_v_gain.weight", "loop_v_forget.weight",
    "loop_mlp_down.weight", "loop_mlp_gain.weight", "loop_mlp_forget.weight",
}


class Plan:
    """The pack's tensors in source order: `rows` = [(GGUF name, source path, absolute offset, bytes, TensorInfo)]."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        g = GGUFFile(path)
        self.kv = {k: _json_safe(v) for k, v in g.metadata.items()}
        if self.kv.get("general.architecture") != ARCH:
            raise S.ConvertError(f"{path.name}: architecture is {self.kv.get('general.architecture')!r}, not {ARCH!r}")
        self.n_layers = self._n_layers(g)
        self.rows, problems = [], []
        seen = set()
        for t in g.tensors:
            if t.name not in GLOBAL and not self._layer_name(t.name):
                problems.append(f"{t.name}: not a tensor of an {ARCH} model")
                continue
            size = t.expected_bytes()
            if size is None:
                problems.append(f"{t.name}: {t.type_name} {list(t.shape)} is not whole blocks")
                continue
            seen.add(t.name)
            self.rows.append((t.name, path, g.data_start + t.offset, size, t))
        for il in range(self.n_layers):
            for base in sorted(self._required(il)):
                if f"blk.{il}.{base}" not in seen:
                    problems.append(f"blk.{il}.{base}: missing")
        for name in sorted(GLOBAL):
            if name not in seen:
                problems.append(f"{name}: missing")
        if problems:
            more = f"\n... and {len(problems) - 8} more" if len(problems) > 8 else ""
            raise S.ConvertError("\n".join(problems[:8]) + more)
        self.rows.sort(key=lambda r: r[2])   # one sequential read of the source

    def _required(self, il: int) -> set:
        """The tensors stored layer `il` must have: full-attention layers have Q/K/V input norms and K/V convolution
        projections in place of attn_norm; layer 0 has no attention-residual query; only the looped core has message
        controllers; NVFP4 experts carry per-expert scales (the MXFP8-era Q8_0 layers do not)."""
        kv = self.kv
        core0, core1 = int(kv.get(A + "loop.core_start", 0)), int(kv.get(A + "loop.core_end", 0))
        n_pass = int(kv.get(A + "num_loops", 1))
        pattern = kv.get(A + "attention.sliding_window_pattern") or []
        visit = il if il < core1 else il + (core1 - core0) * (n_pass - 1)
        swa = bool(pattern[visit]) if visit < len(pattern) else True
        full_only = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
                     "attn_k_conv_proj.weight", "attn_v_conv_proj.weight"}
        req = set(LAYER) - full_only - {"attn_norm.weight"} - {n for n in LAYER if n.startswith("loop_")}
        req -= {"ffn_gate_up_exps.scale", "ffn_down_exps.scale"}
        req |= {"attn_norm.weight"} if swa else full_only
        if il == 0:
            req.discard("attn_res_query.weight")
        if n_pass > 1 and core0 <= il < core1:
            req |= {n for n in LAYER if n.startswith("loop_")}
        return req

    @staticmethod
    def _layer_name(name: str) -> bool:
        m = re.match(r"blk\.\d+\.(.+)$", name)
        return bool(m and m.group(1) in LAYER)

    def _n_layers(self, g) -> int:
        """The physical layers, from the `blk.N.*` tensors. `inkling2.block_count` counts layer visits, not layers."""
        idx = sorted({int(m.group(1)) for t in g.tensors if (m := re.match(r"blk\.(\d+)\.", t.name))})
        if not idx:
            raise S.ConvertError(f"{g.path.name}: no blk.N.* tensors")
        if idx != list(range(len(idx))):
            raise S.ConvertError(f"{g.path.name}: layers are not 0..N-1 ({idx[0]}..{idx[-1]})")
        return len(idx)


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
        die("an inkling2 GGUF is one file; merge split GGUFs first (llama-gguf-split --merge)")
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    mmproj = pathlib.Path(a.mmproj) if a.mmproj and not a.no_vision else None
    if mmproj is not None and not mmproj.is_file():
        die(f"missing input: {mmproj}")
    dspark = pathlib.Path(a.dspark) if getattr(a, "dspark", None) else None
    if dspark is not None and not dspark.is_file():
        die(f"missing input: {dspark}")
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
            if dspark is not None:
                job.file("dspark/gguf", dspark)
            tens = {n: {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.shape],
                        "bytes": size, "offset": off_, "source_sha256": digests[n]}
                    for (n, _, _, size, t_), off_ in zip(plan.rows, offs)}
            lic = str(plan.kv.get("general.license", ""))
            meta = {
                "format": "halo", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py", "path": "gguf_inkling2"},
                "sources": [{"path": src.name, "size": src.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src)}] +
                           ([{"path": mmproj.name, "size": mmproj.stat().st_size,
                              "fingerprint_sha256_first_last_64mib_and_size": fingerprint(mmproj)}] if mmproj else []) +
                           ([{"path": dspark.name, "size": dspark.stat().st_size,
                              "fingerprint_sha256_first_last_64mib_and_size": fingerprint(dspark)}] if dspark else []),
                "gguf_kv": plan.kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic), **({"quantized_by": plan.kv["general.quantized_by"]}
                                                              if plan.kv.get("general.quantized_by") else {})},
                "gguf": {"lossless": "every tensor's bytes as the source GGUF stores them; source_sha256 per tensor",
                         "block_count": f"{plan.n_layers} physical layers (the GGUF's inkling2.block_count "
                                        "counts layer visits)"},
                "vision": {"mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size} if mmproj else None,
                "dspark": {"section": "dspark/gguf", "bytes": dspark.stat().st_size,
                           "sha256": sha256_range(dspark, 0, dspark.stat().st_size)} if dspark else None,
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
    """Every `tokenizer/*` section against the file the converter writes from the GGUF's metadata."""
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


def _verify_dspark(mf, meta: dict, pack: pathlib.Path, dspark) -> list[str]:
    """`dspark/gguf` against meta.dspark's size and SHA-256, and byte for byte against the file when one is given."""
    bad = []
    names = mf.names()
    rec = meta.get("dspark")
    has = "dspark/gguf" in names
    if has and not isinstance(rec, dict):
        bad.append("meta dspark: the pack has dspark/gguf, meta has no dspark record")
    if rec and not has:
        bad.append("dspark/gguf: meta names it, the pack has no such section")
    if has:
        off, size = mf.section("dspark/gguf")
        if isinstance(rec, dict):
            if rec.get("bytes") != size:
                bad.append(f"meta dspark.bytes {rec.get('bytes')}, the section is {size:,} bytes")
            if rec.get("section") != "dspark/gguf":
                bad.append(f"meta dspark.section {rec.get('section')!r}, the section is 'dspark/gguf'")
            got_h = sha256_range(pack, off, size)
            if rec.get("sha256") != got_h:
                bad.append(f"dspark/gguf: SHA-256 {got_h[:16]} vs meta {str(rec.get('sha256'))[:16]}")
        if dspark is not None:
            src_n = os.path.getsize(dspark)
            src_h = sha256_range(dspark, 0, src_n)
            pack_h = sha256_range(pack, off, size)
            if not (src_h == pack_h and src_n == size):
                bad.append(f"dspark/gguf: {size:,} bytes in the pack, {src_n:,} in {dspark.name} "
                           f"(SHA-256 {pack_h[:16]} vs {src_h[:16]})")
    elif dspark is not None:
        bad.append(f"dspark/gguf: --dspark {dspark.name} was given, the pack has no such section")
    return bad


def verify(pack: pathlib.Path, gguf: pathlib.Path, mmproj=None, tokenizer_files=None, dspark=None) -> list[str]:
    """Everything of `pack` this GGUF and projector would produce, against what they are now:

      tensors        every tensor's type, shape, bytes and SHA-256 against the GGUF's, and the meta's record of each
      meta           gguf_kv against the GGUF's metadata, sources[0] against the GGUF's name and size
      tokenizer/     each section byte for byte against what convert.py's export writes from the GGUF
      vision/        mmproj byte for byte against `mmproj` when given, text_vocab against the vocab-only GGUF, and
                     meta.vision against the two sections
      dspark/gguf    its size and SHA-256 against meta.dspark, and byte for byte against `dspark` when given

    Returns the mismatches; empty when the pack is what these inputs produce. `tokenizer_files` replaces the converter's
    export (tests use it); `mmproj` and `dspark` may be None, and then those sections are not compared against a file."""
    mf = ModelFile(str(pack))
    meta = json.loads(mf.read("meta"))
    plan = Plan(gguf)
    bad = _verify_tensors(mf, meta, plan, pack, gguf)
    bad += _verify_meta(meta, plan, gguf)
    bad += _verify_tokenizer(mf, gguf, tokenizer_files or _export_tokenizer)
    bad += _verify_vision(mf, meta, pack, gguf, mmproj)
    bad += _verify_dspark(mf, meta, pack, dspark)
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check an inkling2 pack against the GGUF (and projector, drafter) it was "
                                             "converted from, section by section.")
    ap.add_argument("cmd", choices=("verify",))
    ap.add_argument("pack")
    ap.add_argument("gguf")
    ap.add_argument("--mmproj", metavar="GGUF", help="the projector GGUF the pack was built with; its bytes are "
                    "compared with vision/mmproj")
    ap.add_argument("--dspark", metavar="GGUF", help="the DSpark drafter GGUF the pack was built with; its bytes are "
                    "compared with dspark/gguf")
    a = ap.parse_args()
    pack, gguf = pathlib.Path(a.pack), pathlib.Path(a.gguf)
    bad = verify(pack, gguf, pathlib.Path(a.mmproj) if a.mmproj else None,
                 dspark=pathlib.Path(a.dspark) if a.dspark else None)
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
    if "dspark/gguf" in mf.names() and not a.dspark:
        print("note: dspark/gguf is not compared: pass --dspark to check it")
    print(f"verify: {len(meta['tensors'])} tensors, {n_tok} tokenizer files, {len(vis)} vision sections, "
          f"{len(bad)} mismatches -> {'LOSSLESS' if not bad else 'FAIL'}")
    sys.exit(1 if bad else 0)
