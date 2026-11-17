"""The release archive carries bin/halo as a real symlink to bin/halocline on Linux.

    python3 -m unittest tools.test_package_release

The tree is synthetic: the test writes the files the script requires, packs them, and reads the archive's member table.
No build and no model.
"""
import contextlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.package_release as pkg

# the names after the rename; the script's own EXECUTABLES change to these in the same release
RENAMED = ("halocline", "halocline-bench", "halocline-probe", "halocline-vision")


class PackageRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve() / "tree"
        self.root.mkdir()
        (self.root / "CMakeLists.txt").write_text("project(halocline VERSION 0.3.2 LANGUAGES C CXX CUDA)\n",
                                                  encoding="utf-8")
        for rel in pkg.FILES + pkg.TOOLS:
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n", encoding="utf-8")
        for rel, body in (("serve/server.py", ""), ("data/profiles/p.prof", ""),
                          ("data/bench/conv.ids", "1,2\n"), ("docs/index.md", "# docs\n")):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        self.out = Path(self.tmp.name).resolve() / "dist"

    def tearDown(self):
        self.tmp.cleanup()

    def binaries(self, names):
        (self.root / "bin").mkdir()
        for name in names:
            path = self.root / "bin" / name
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o755)

    def pack(self):
        argv = ["package_release.py", "--tag", "v0.3.2", "--platform", "linux-x64",
                "--root", str(self.root), "--out", str(self.out)]
        printed = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
            self.assertEqual(pkg.main(), 0)
        archive = self.out / "halo-v0.3.2-linux-x64.tar.gz"
        self.assertEqual(printed.getvalue().strip(), str(archive))
        return archive

    def test_linux_archive_links_halo_to_halocline(self):
        self.binaries(RENAMED)
        with mock.patch.object(pkg, "EXECUTABLES", RENAMED):
            archive = self.pack()
        with tarfile.open(archive) as t:
            top = "halo-v0.3.2-linux-x64"
            link = t.getmember(f"{top}/bin/halo")
            self.assertTrue(link.issym())
            self.assertEqual(link.linkname, "halocline")
            self.assertTrue(t.getmember(f"{top}/bin/halocline").isfile())
            # one member only: the alias is not also copied as a file
            self.assertEqual(sum(1 for m in t.getmembers() if m.name == f"{top}/bin/halo"), 1)

    def test_pre_rename_tree_still_packs(self):
        self.binaries(("halo", "halo-bench", "halo-probe", "halo-vision"))
        archive = self.pack()
        with tarfile.open(archive) as t:
            top = "halo-v0.3.2-linux-x64"
            self.assertTrue(t.getmember(f"{top}/bin/halo").isfile())
            self.assertNotIn(f"{top}/bin/halocline", [m.name for m in t.getmembers()])


if __name__ == "__main__":
    unittest.main()
