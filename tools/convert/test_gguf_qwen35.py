"""Tests for gguf_qwen35.py (a qwen35 pack from a llama.cpp GGUF, every tensor copied as stored) on a tiny synthetic
model: four main layers (three GDN, one attention) and the drafter layer llama.cpp stores as blk.4.

    python3 -m unittest discover -s tools/convert -p 'test_gguf_qwen35.py'"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert  # noqa: E402
import gguf_qwen35 as G  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402

H, FF, N_LAYERS = 256, 512, 4
# bytes per 256 values of the block types used here (ggml's block sizes)
PER_256 = {"Q8_0": 272, "Q4_K": 144, "Q6_K": 210, "IQ4_XS": 136}


class TinyModel:
    """A synthetic qwen35 model: four main layers (three GDN, one attention) and the drafter layer llama.cpp stores as
    blk.4. A fixture rather than a test case, so the pack-integrity tests build the same model."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.rng = np.random.default_rng(0)
        self.src = {}
        self.gguf = self.write_gguf()

    def tearDown(self):
        self.tmp.cleanup()

    def quant(self, g, name, cols, rows, ty):
        data = self.rng.bytes(cols * rows // 256 * PER_256[ty])
        g.add_raw(name, [cols, rows], ty, data)
        self.src[name] = data

    def f32(self, g, name, n):
        a = self.rng.standard_normal(n).astype(np.float32)
        g.add_f32(name, a)
        self.src[name] = a.tobytes()

    def write_gguf(self, drop=(), nextn=1):
        g = GGUFWriter()
        g.add("general.architecture", "qwen35")
        g.add("general.basename", "Tiny")
        g.add("general.license", "apache-2.0")
        g.add("general.quantized_by", "Unsloth")
        g.add("qwen35.block_count", N_LAYERS + nextn, "u32")
        g.add("qwen35.nextn_predict_layers", nextn, "u32")
        g.add("qwen35.full_attention_interval", 4, "u32")
        g.add("qwen35.embedding_length", H, "u32")
        g.add("tokenizer.ggml.model", "gpt2")
        g.add("tokenizer.ggml.tokens", ["a", "b", "c", "d"])
        g.add("tokenizer.ggml.merges", ["a b"])
        g.add("tokenizer.ggml.token_type", [1, 1, 1, 1])
        g.add("tokenizer.chat_template", "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}")
        put = {"Q": self.quant, "F": lambda g_, n, c, r, ty: self.f32(g_, n, c * r)}
        tensors = [("token_embd.weight", 256, 4, "Q4_K"), ("output.weight", 256, 4, "Q6_K"), ("output_norm.weight", H, 1, "F")]
        for il in range(N_LAYERS + nextn):
            p = f"blk.{il}."
            tensors += [(p + "attn_norm.weight", H, 1, "F"), (p + "post_attention_norm.weight", H, 1, "F"),
                        (p + "ffn_gate.weight", H, FF, "IQ4_XS"), (p + "ffn_up.weight", H, FF, "Q4_K"),
                        (p + "ffn_down.weight", FF, H, "Q6_K")]
            if il < N_LAYERS and (il + 1) % 4 != 0:
                tensors += [(p + "attn_qkv.weight", H, 512, "IQ4_XS"), (p + "attn_gate.weight", H, 256, "Q4_K"),
                            (p + "ssm_out.weight", 256, H, "Q6_K"), (p + "ssm_alpha.weight", H, 8, "Q8_0"),
                            (p + "ssm_beta.weight", H, 8, "Q8_0"), (p + "ssm_a", 8, 1, "F"), (p + "ssm_dt.bias", 8, 1, "F"),
                            (p + "ssm_conv1d.weight", 4, 512, "F"), (p + "ssm_norm.weight", 128, 1, "F")]
            else:
                tensors += [(p + "attn_q.weight", H, 512, "Q6_K"), (p + "attn_k.weight", H, 256, "Q8_0"),
                            (p + "attn_v.weight", H, 256, "Q8_0"), (p + "attn_output.weight", 256, H, "Q4_K"),
                            (p + "attn_q_norm.weight", 256, 1, "F"), (p + "attn_k_norm.weight", 256, 1, "F")]
            if il == N_LAYERS:
                tensors += [(p + "nextn.eh_proj.weight", 2 * H, H, "Q6_K"), (p + "nextn.enorm.weight", H, 1, "F"),
                            (p + "nextn.hnorm.weight", H, 1, "F"), (p + "nextn.shared_head_norm.weight", H, 1, "F")]
        for name, c, r, ty in tensors:
            if name not in drop:
                put["F" if ty == "F" else "Q"](g, name, c, r, ty)
        return g.write(self.t / "tiny-UD-IQ4_XS.gguf")

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf)], out=str(self.t / "out.halo"), mmproj=None, no_vision=True, name=None,
                 force=False, dry_run=False)
        d.update(kw)
        return type("A", (), d)()

    @staticmethod
    def stub_tokenizer(src, out_dir):
        """A stand-in for convert.export_tokenizer that needs no regex: one file, written the same way each time."""
        d = Path(out_dir) / "tokenizer"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokenizer.json").write_bytes(b"{}")
        return sorted(d.iterdir())

    def build(self, export=None, **kw):
        def die(msg):
            raise S.ConvertError(msg)
        a = self.args(**kw)
        G.build(a, convert.Job, convert.fingerprint, die, export or self.stub_tokenizer)
        mf = ModelFile(str(a.out))
        return mf, json.loads(mf.read("meta"))

    def flip(self, section, offset=0):
        """Flip one bit of one byte inside `section` of out.halo; the directory keeps the hash it had."""
        mf = ModelFile(str(self.t / "out.halo"))
        off, _ = mf.section(section)
        with open(self.t / "out.halo", "r+b") as f:
            f.seek(off + offset)
            b = f.read(1)
            f.seek(off + offset)
            f.write(bytes([b[0] ^ 1]))


class Qwen35GgufTest(TinyModel, unittest.TestCase):
    def test_every_tensor_is_copied_as_stored(self):
        mf, meta = self.build()
        gguf = mf.read("gguf")
        names = {n if not n.startswith(f"blk.{N_LAYERS}.") else G.MTP_NAMES[n.split(".", 2)[2]]: n for n in self.src}
        self.assertEqual(set(meta["tensors"]), set(names))
        for pack_name, src_name in names.items():
            e = meta["tensors"][pack_name]
            got = gguf[e["offset"]:e["offset"] + e["bytes"]]
            self.assertEqual(got, self.src[src_name], pack_name)
            self.assertEqual(e["source_sha256"], hashlib.sha256(self.src[src_name]).hexdigest(), pack_name)
            self.assertEqual(e["offset"] % 32, 0)
        self.assertEqual(meta["tensors"]["blk.0.ffn_gate.weight"]["type"], "IQ4_XS")
        self.assertEqual(meta["tensors"]["mtp.fc.weight"]["shape"], [2 * H, H])
        self.assertEqual(G.verify(self.t / "out.halo", self.gguf, tokenizer_files=self.stub_tokenizer), [])

    def test_meta_counts_main_layers_and_names_the_drafter(self):
        _, meta = self.build()
        self.assertEqual(meta["arch"], "qwen35")
        self.assertEqual(meta["gguf_kv"]["qwen35.block_count"], N_LAYERS)
        self.assertEqual(meta["gguf_kv"]["qwen35.nextn_predict_layers"], 1)
        self.assertEqual(meta["name"], "Tiny UD-IQ4_XS")
        self.assertEqual(meta["license"], {"name": "Apache License 2.0", "quantized_by": "Unsloth"})
        self.assertEqual(meta["gguf"]["renamed"]["blk.4.nextn.eh_proj.weight"], "mtp.fc.weight")
        self.assertIsNotNone(meta["mtp"])
        self.assertFalse(any(n.startswith("blk.4.") for n in meta["tensors"]))

    def test_without_drafter(self):
        self.src.clear()
        self.gguf = self.write_gguf(nextn=0)
        _, meta = self.build(force=True)
        self.assertEqual(meta["gguf_kv"]["qwen35.block_count"], N_LAYERS)
        self.assertIsNone(meta["mtp"])
        self.assertEqual(meta["gguf"]["renamed"], {})

    def test_missing_tensor_is_refused(self):
        self.src.clear()
        self.gguf = self.write_gguf(drop=("blk.3.attn_k.weight", "blk.4.nextn.enorm.weight"))
        with self.assertRaises(S.ConvertError) as cm:
            G.Plan(self.gguf)
        self.assertIn("blk.3.attn_k.weight: missing", str(cm.exception))
        self.assertIn("mtp.pre_fc_norm_embedding.weight: missing", str(cm.exception))

    def test_other_architecture_is_refused(self):
        g = GGUFWriter()
        g.add("general.architecture", "qwen4exp")
        p = g.write(self.t / "other.gguf")
        with self.assertRaises(S.ConvertError):
            G.Plan(p)

    def test_verify_sees_a_changed_byte(self):
        self.build()
        mf = ModelFile(str(self.t / "out.halo"))
        meta = json.loads(mf.read("meta"))
        base, _ = mf.section("gguf")
        e = meta["tensors"]["blk.2.ssm_out.weight"]
        with open(self.t / "out.halo", "r+b") as f:
            f.seek(base + e["offset"] + 5)
            b = f.read(1)
            f.seek(base + e["offset"] + 5)
            f.write(bytes([b[0] ^ 1]))
        bad = G.verify(self.t / "out.halo", self.gguf, tokenizer_files=self.stub_tokenizer)
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("blk.2.ssm_out.weight: SHA-256"))


if __name__ == "__main__":
    unittest.main()
