"""Tests for depth_kld.py: base headers and sizes (stock and depth), and the per-depth plan.

    python3 -m unittest discover -s tools/depth_kld -p 'test_*.py'"""
import io
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_kld as D  # noqa: E402


def write_base(path, n_ctx, n_vocab, n_chunk, first=None, extra=0):
    """A base of the right layout (zero rows): stock when `first` is None, else depth."""
    depth = first is not None
    f0 = first if depth else n_ctx // 2
    with open(path, "wb") as f:
        f.write(D.MAGIC_DEPTH if depth else D.MAGIC_STOCK)
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        if depth:
            f.write(struct.pack("<i", first))
        f.write(b"\0" * (4 * n_ctx * n_chunk + n_chunk * (n_ctx - 1 - f0) * D.row_bytes(n_vocab) + extra))


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def quiet_check(self, *paths):
        with redirect_stdout(io.StringIO()) as out:
            rc = D.check([str(p) for p in paths])
        return rc, out.getvalue()

    def test_row_bytes_is_llama_cpp_nv(self):
        self.assertEqual(D.row_bytes(248320), 2 * (248320 + 4))
        self.assertEqual(D.row_bytes(7), 2 * (8 + 4))     # an odd vocabulary is padded to even

    def test_stock_base(self):
        write_base(self.t / "s.kld", 512, 7, 3)
        h = D.read_header(self.t / "s.kld")
        self.assertEqual((h["depth"], h["first"], h["window"]), (False, 256, 255))
        self.assertEqual(self.quiet_check(self.t / "s.kld")[0], 0)

    def test_depth_base(self):
        write_base(self.t / "d.kld", 600, 7, 2, first=500)
        h = D.read_header(self.t / "d.kld")
        self.assertEqual((h["depth"], h["first"], h["window"]), (True, 500, 99))
        rc, out = self.quiet_check(self.t / "d.kld")
        self.assertEqual(rc, 0)
        self.assertIn("depth base", out)

    def test_size_mismatch_fails(self):
        write_base(self.t / "d.kld", 600, 7, 2, first=500, extra=2)
        rc, out = self.quiet_check(self.t / "d.kld")
        self.assertEqual(rc, 1)
        self.assertIn("expected", out)

    def test_bad_headers(self):
        (self.t / "x.kld").write_bytes(b"notabase" + b"\0" * 32)
        with self.assertRaises(ValueError):
            D.read_header(self.t / "x.kld")
        write_base(self.t / "f.kld", 600, 7, 2, first=599)   # first past n_ctx - 2
        with self.assertRaises(ValueError):
            D.read_header(self.t / "f.kld")


class PlanTest(unittest.TestCase):
    def test_window_after_the_prefix(self):
        s = D.Step(65536, 512, 8, Path("/out"))
        self.assertEqual(s.n_ctx, 65536 + 512 + 1)
        self.assertEqual(s.bytes(), 24 + 4 * s.n_ctx * 8 + 8 * 512 * D.row_bytes(D.N_VOCAB))
        self.assertAlmostEqual(s.kv_gib(), s.n_ctx * 65536 / D.GIB)

    def test_deepest_window_fits_the_context(self):
        s = D.Step(D.MODEL_CONTEXT - 513, 512, 8, Path("/out"))
        self.assertEqual(s.n_ctx, D.MODEL_CONTEXT)
        with self.assertRaises(ValueError) as cm:
            D.Step(D.MODEL_CONTEXT - 512, 512, 8, Path("/out"))
        self.assertIn(str(D.MODEL_CONTEXT - 513), str(cm.exception))

    def test_auto_ngl_leaves_room_for_the_cache(self):
        shallow, deep = D.Step(65536, 512, 8, Path("/o")), D.Step(261631, 512, 8, Path("/o"))
        self.assertGreater(shallow.auto_ngl(), deep.auto_ngl())
        for s in (shallow, deep):
            self.assertLessEqual(s.kv_gib() + s.auto_ngl() * D.LAYER_GIB + D.COMPUTE_GIB, D.CARD_GIB)

    def test_argv(self):
        s = D.Step(140288, 512, 8, Path("/o"))
        argv = s.argv(Path("/llama/bin"), Path("/m/BF16-00001-of-00002.gguf"), Path("/c.txt"), 4096, 8, 16)
        self.assertEqual(argv[0], "/llama/bin/llama-perplexity")
        for flag, value in (("-c", str(140288 + 513)), ("--chunks", "8"), ("-ub", "4096"), ("-fa", "on"), ("-ngl", "8"),
                            ("--kl-divergence-base", "/o/qwen38-27b-bf16-depth140288-w512-c8.kld")):
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_plan_prints_one_command_per_depth(self):
        a = D.argparse.Namespace(out_dir="/o", depths="65536,261631", window=512, chunks=8, model="/m.gguf", text="/c.txt",
                                 llama_bin="/b", batch=4096, ngl=-1, threads=16, run=False, force=False)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(D.plan(a), 0)
        lines = [x for x in out.getvalue().splitlines() if x.startswith("LLAMA_KL_FIRST=")]
        self.assertEqual([x.split()[0] for x in lines], ["LLAMA_KL_FIRST=65536", "LLAMA_KL_FIRST=261631"])
        a.depths = "262000"
        with redirect_stdout(io.StringIO()), redirect_stdout(io.StringIO()):
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(D.plan(a), 2)


if __name__ == "__main__":
    unittest.main()
