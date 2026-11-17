"""`halocline serve` hands the Python server what was typed, with only a leading MODEL.halo turned into --model.

    HALO_BIN=bin/halocline python3 -m unittest tools.test_serve_launcher
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("HALO_BIN", ROOT / "bin" / "halocline")).resolve()


@unittest.skipUnless(ENGINE.is_file(), "build bin/halocline first")
class ServeLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name).resolve()
        self.dir = d
        (d / "m.halo").write_text("")
        (d / "c.json").write_text("{}")
        self.out = d / "argv.txt"
        stub = d / "python"                            # records its arguments, one per line
        stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{self.out}"\n')
        stub.chmod(0o755)
        self.stub = stub

    def tearDown(self):
        self.tmp.cleanup()

    def served(self, *args: str) -> list[str]:
        r = subprocess.run([str(ENGINE), "serve", *args, "--python", str(self.stub)], capture_output=True, text=True, cwd=self.dir)
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = self.out.read_text().splitlines()
        self.assertEqual(argv[:2], ["-m", "serve.server"])
        self.assertEqual(argv[2], "--engine-bin")
        # the engine the server starts is the real binary, even when this one was reached through the bin/halo symlink
        self.assertEqual(Path(argv[3]).name, "halocline", argv[3])
        return argv[4:]

    def test_option_values_pass_through(self):
        self.assertEqual(self.served("--config", "c.json", "--parallel", "4"), ["--config", str(self.dir / "c.json"), "--parallel", "4"])

    def test_first_bare_argument_is_the_model(self):
        self.assertEqual(self.served("m.halo", "--parallel", "4"), ["--model", str(self.dir / "m.halo"), "--parallel", "4"])

    def test_later_bare_word_is_not_a_model(self):
        self.assertEqual(self.served("--max-context", "4096", "--parallel", "4", "--config", "c.json"),
                         ["--max-context", "4096", "--parallel", "4", "--config", str(self.dir / "c.json")])


if __name__ == "__main__":
    unittest.main()
