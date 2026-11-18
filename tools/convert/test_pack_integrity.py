"""Pack integrity on the synthetic qwen35 model of test_gguf_qwen35.py:

  tools/model_file.py verify   every section against the SHA-256 in the directory, the header checks the engine makes
                               (magic, version, section count, file size, section bounds and alignment), streaming
  gguf_qwen35.py verify        tensors, meta's gguf_kv and sources, tokenizer/ against what the converter writes from
                               the GGUF, and the vision sections against the projector and the vocab-only GGUF
  reproducibility              two conversions of one GGUF are the same bytes

    python3 -m unittest discover -s tools/convert -p 'test_pack_integrity.py'"""
import contextlib
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert  # noqa: E402
import gguf_qwen35 as G  # noqa: E402
import model_file as M  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ENTRY, HEADER, HEADER_BYTES, LEGACY_MAGIC, ModelFile  # noqa: E402
from test_gguf_qwen35 import TinyModel  # noqa: E402

HAVE_REGEX = importlib.util.find_spec("regex") is not None


class PackVerifyTest(TinyModel, unittest.TestCase):
    def cli(self, *argv):
        """model_file.main in this process, with stdout and stderr kept: (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = M.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def line_for(self, out, name):
        return next(l for l in out.splitlines() if l.startswith(f"  {name} ") and " bytes  " in l)

    def test_every_section_verifies(self):
        mf, _ = self.build()
        code, out, _ = self.cli("verify", str(self.t / "out.halo"))
        self.assertEqual(code, 0, out)
        self.assertEqual(sum(1 for l in out.splitlines() if l.startswith("  ") and " bytes  " in l), len(mf.names()))
        for name in mf.names():
            self.assertIn("ok", self.line_for(out, name))
        self.assertIn(f"{len(mf.names())} sections", out)
        self.assertIn("-> OK", out)
        size = os.path.getsize(self.t / "out.halo")
        self.assertEqual(out.splitlines()[0], f"{self.t / 'out.halo'}: {size:,} bytes ({size / (1 << 30):.2f} GiB)")

    def test_a_flipped_tensor_byte_is_caught(self):
        _, meta = self.build()
        self.flip("gguf", meta["tensors"]["blk.0.ffn_up.weight"]["offset"] + 3)
        code, out, _ = self.cli("verify", str(self.t / "out.halo"))
        self.assertEqual(code, 1)
        self.assertIn("BAD", self.line_for(out, "gguf"))
        self.assertIn("SHA-256", self.line_for(out, "gguf"))
        self.assertIn("ok", self.line_for(out, "meta"))
        self.assertIn("-> FAIL", out)

    def test_a_flipped_meta_byte_is_caught(self):
        self.build()
        self.flip("meta", 5)
        code, out, _ = self.cli("verify", str(self.t / "out.halo"))
        self.assertEqual(code, 1)
        self.assertIn("BAD", self.line_for(out, "meta"))
        self.assertIn("ok", self.line_for(out, "gguf"))

    def test_a_flipped_tokenizer_byte_is_caught(self):
        self.build()
        self.flip("tokenizer/tokenizer.json")
        code, out, _ = self.cli("verify", str(self.t / "out.halo"))
        self.assertEqual(code, 1)
        self.assertIn("BAD", self.line_for(out, "tokenizer/tokenizer.json"))
        self.assertIn("ok", self.line_for(out, "meta"))

    def test_section_filter_checks_only_that_section(self):
        _, meta = self.build()
        self.flip("gguf", meta["tensors"]["blk.1.ffn_up.weight"]["offset"])
        code, out, _ = self.cli("verify", str(self.t / "out.halo"), "--section", "meta")
        self.assertEqual(code, 0, out)
        self.assertIn("1 of 3 sections", out)
        code, out, _ = self.cli("verify", str(self.t / "out.halo"), "--section", "gguf", "--section", "meta")
        self.assertEqual(code, 1)
        self.assertIn("2 of 3 sections", out)

    def test_unknown_section_is_a_usage_error(self):
        self.build()
        code, _, err = self.cli("verify", str(self.t / "out.halo"), "--section", "nope")
        self.assertEqual(code, 2)
        self.assertIn("no section named nope", err)
        self.assertIn("gguf", err)

    def test_truncated_file_is_a_header_failure(self):
        self.build()
        p = self.t / "out.halo"
        p.write_bytes(p.read_bytes()[:-1])
        code, out, _ = self.cli("verify", str(p))
        self.assertEqual(code, 1)
        self.assertIn("header:", out)
        self.assertIn("truncated", out)
        self.assertIn("verify: FAIL", out)

    def test_bad_magic_is_a_header_failure(self):
        self.build()
        p = self.t / "out.halo"
        with open(p, "r+b") as f:
            f.seek(0)
            f.write(b"NOTAPACK")
        code, out, _ = self.cli("verify", str(p))
        self.assertEqual(code, 1)
        self.assertIn("BAD", out)
        self.assertIn("magic", out)

    def test_the_legacy_magic_verifies(self):
        self.build()
        p = self.t / "out.halo"
        with open(p, "r+b") as f:
            f.seek(0)
            f.write(LEGACY_MAGIC)
        code, out, _ = self.cli("verify", str(p))
        self.assertEqual(code, 0, out)
        self.assertIn("the spelling files written before the rename carry", out)

    def test_a_file_shorter_than_the_header_is_a_header_failure(self):
        p = self.t / "short.halo"
        p.write_bytes(b"\0" * 100)
        code, out, _ = self.cli("verify", str(p))
        self.assertEqual(code, 1)
        self.assertIn(f"{HEADER_BYTES}-byte header", out)

    def test_a_section_entry_outside_the_file_is_caught(self):
        self.build()
        p = self.t / "out.halo"
        with open(p, "r+b") as f:
            head = f.read(HEADER_BYTES)
            # entry 1 (gguf): push its offset past the declared end
            name, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + ENTRY.size)
            file_bytes = HEADER.unpack_from(head, 0)[3]
            patched = ENTRY.pack(name, file_bytes + 2 << 20, size, digest, b"\0" * 16)
            f.seek(HEADER.size + ENTRY.size)
            f.write(patched)
        code, out, _ = self.cli("verify", str(p))
        self.assertEqual(code, 1)
        self.assertIn("outside the declared file", out)

    def test_the_script_is_a_command(self):
        self.build()
        r = subprocess.run([sys.executable, str(Path(M.__file__)), "verify", str(self.t / "out.halo")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("-> OK", r.stdout)

    def test_two_conversions_are_byte_identical(self):
        """The same GGUF, converted twice into two files: the whole file, header, directory and every section."""
        self.build(out=str(self.t / "a.halo"))
        self.build(out=str(self.t / "b.halo"))
        a, b = (self.t / "a.halo").read_bytes(), (self.t / "b.halo").read_bytes()
        ha, hb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
        self.assertEqual(ha, hb)


@unittest.skipUnless(HAVE_REGEX, "tools/tokenizer.py needs the regex module")
class GgufVerifyTest(TinyModel, unittest.TestCase):
    """gguf_qwen35.verify with the real tokenizer export (convert.export_tokenizer) and the vision inputs."""

    def build_real(self, **kw):
        return self.build(export=convert.export_tokenizer, **kw)

    def mmproj(self):
        g = GGUFWriter()
        g.add("general.architecture", "clip")
        g.add("clip.projector_type", "qwen3vl")
        g.add("tokenizer.ggml.tokens", ["a", "b"])
        return g.write(self.t / "mmproj.gguf")

    def test_tensor_tokenizer_and_template_sections_match(self):
        mf, _ = self.build_real()
        self.assertIn("tokenizer/chat_template.jinja", mf.names())
        self.assertIn("tokenizer/merges.txt", mf.names())
        self.assertEqual(G.verify(self.t / "out.halo", self.gguf), [])

    def test_a_flipped_tokenizer_byte_is_caught(self):
        self.build_real()
        self.flip("tokenizer/vocab.json", 2)
        bad = G.verify(self.t / "out.halo", self.gguf)
        self.assertEqual(len(bad), 1, bad)
        self.assertTrue(bad[0].startswith("tokenizer/vocab.json:"), bad[0])

    def test_a_changed_meta_key_is_caught(self):
        mf, _ = self.build_real()
        raw = mf.read("meta")
        before, after = b'"qwen35.embedding_length":256', b'"qwen35.embedding_length":257'
        self.assertIn(before, raw)
        off, _ = mf.section("meta")
        with open(self.t / "out.halo", "r+b") as f:
            f.seek(off + raw.index(before))
            f.write(after)
        bad = G.verify(self.t / "out.halo", self.gguf)
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("meta gguf_kv", bad[0])
        self.assertIn("qwen35.embedding_length", bad[0])

    def test_vision_sections_match_the_inputs(self):
        mmproj = self.mmproj()
        mf, meta = self.build_real(mmproj=str(mmproj), no_vision=False)
        self.assertEqual(meta["vision"]["mmproj_bytes"], mf.section("vision/mmproj")[1])
        self.assertEqual(G.verify(self.t / "out.halo", self.gguf, mmproj=mmproj), [])
        self.assertEqual(G.verify(self.t / "out.halo", self.gguf), [])   # not compared, not failed

    def test_a_changed_projector_is_caught(self):
        mmproj = self.mmproj()
        self.build_real(mmproj=str(mmproj), no_vision=False)
        self.flip("vision/mmproj", 1)
        bad = G.verify(self.t / "out.halo", self.gguf, mmproj=mmproj)
        self.assertEqual(len(bad), 1, bad)
        self.assertTrue(bad[0].startswith("vision/mmproj:"), bad[0])

    def test_a_changed_text_vocab_is_caught(self):
        self.build_real(mmproj=str(self.mmproj()), no_vision=False)
        self.flip("vision/text_vocab", 3)
        bad = G.verify(self.t / "out.halo", self.gguf)
        self.assertEqual(len(bad), 1, bad)
        self.assertTrue(bad[0].startswith("vision/text_vocab:"), bad[0])

    def test_a_projector_the_pack_does_not_carry_is_caught(self):
        self.build_real()
        bad = G.verify(self.t / "out.halo", self.gguf, mmproj=self.mmproj())
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("the pack has no such section", bad[0])

    def test_two_conversions_with_the_real_tokenizer_are_byte_identical(self):
        self.build_real(out=str(self.t / "a.halo"))
        self.build_real(out=str(self.t / "b.halo"))
        self.assertEqual(hashlib.sha256((self.t / "a.halo").read_bytes()).hexdigest(),
                         hashlib.sha256((self.t / "b.halo").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
