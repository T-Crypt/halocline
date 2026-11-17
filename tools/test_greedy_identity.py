"""The greedy identity gate's argument handling on a dense and a sparse pack.

    python3 -m unittest tools.test_greedy_identity

Synthetic packs only: no engine, no GPU and no model weights.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.greedy_identity as gi
import model_file  # noqa: E402  (tools/ is on sys.path after the import above)


class GreedyIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def pack(self, arch: str) -> Path:
        path = self.dir / f"{arch or 'unknown'}.halo"
        w = model_file.ModelWriter(str(path))
        w.add_bytes("meta", json.dumps({"arch": arch}).encode())
        w.close()
        return path

    def test_dense_pack_pins_no_expert_flags(self):
        pack = self.pack("qwen35")
        self.assertEqual(gi.pinned_flags(str(pack)), [])
        args = gi.engine_args(str(pack), ["--spec", "8"])
        self.assertEqual(args, ["--model", str(pack), "--spec", "8", "--max-context", "8192"])
        self.assertNotIn("--adapt-swaps", args)

    def test_sparse_pack_pins_adaptive_swaps_and_keeps_the_tier(self):
        pack = self.pack("qwen4exp")
        self.assertEqual(gi.pinned_flags(str(pack)), ["--adapt-swaps", "0"])
        self.assertEqual(gi.engine_args(str(pack), ["--spec", "0", "--expert-cache", "7000"]),
                         ["--model", str(pack), "--spec", "0", "--max-context", "8192",
                          "--adapt-swaps", "0", "--expert-cache", "7000"])

    def test_set_args_replaces_a_value_and_appends_a_bare_flag(self):
        self.assertEqual(gi.set_args(["--a", "1", "--b"], ["--a", "2", "--c"]), ["--a", "2", "--b", "--c"])
        self.assertEqual(gi.set_args(["--greedy", "--x", "1"], ["--greedy"]), ["--greedy", "--x", "1"])

    def test_a_file_that_is_not_a_pack_is_refused_before_the_engine(self):
        for name in gi.PROMPTS:
            (self.dir / f"{name}.ids").write_text("1,2\n", encoding="utf-8")
        bad = self.dir / "m.halo"
        bad.write_text("not a pack", encoding="utf-8")
        argv = ["greedy_identity.py", "ref", "new", "--model", str(bad), "--fixtures", str(self.dir)]
        err = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err):
            self.assertEqual(gi.main(), 2)
        self.assertIn("cannot read", err.getvalue())


if __name__ == "__main__":
    unittest.main()
