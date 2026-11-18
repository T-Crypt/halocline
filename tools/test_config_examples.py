"""The sample configs stay true to the docs: the root halo_config.json is the sparse example of
docs/configuration/examples.md, and every example there and in examples/ loads through the server's own config reader
with .halo packs (2026-11-18: the root file still named a pre-rename .basalt pack, on Strata's old settings)."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.config import build_parser, load_config_file  # noqa: E402

DOC = ROOT / "docs/configuration/examples.md"


def doc_blocks():
    """The JSON code blocks of the examples page, in order."""
    return [json.loads(b) for b in re.findall(r"```json\n(.*?)```", DOC.read_text(), re.S)]


def files():
    return [ROOT / "halo_config.json", *sorted((ROOT / "examples").glob("*.json"))]


class ConfigExamples(unittest.TestCase):
    def test_root_sample_is_the_documented_sparse_example(self):
        self.assertEqual(json.loads((ROOT / "halo_config.json").read_text()), doc_blocks()[0])

    def test_every_example_names_a_halo_pack(self):
        for cfg in [*doc_blocks(), *(json.loads(p.read_text()) for p in files())]:
            self.assertEqual(cfg["version"], 2)
            for m in cfg["models"]:
                self.assertTrue(m["model"].endswith(".halo"), m["model"])

    def test_every_example_file_loads(self):
        for p in files():
            with self.subTest(p.name):
                cfg = load_config_file(str(p), build_parser())
                self.assertGreaterEqual(len(cfg.models), 1)

    def test_the_dense_default_is_the_long_context_profile(self):
        default = json.loads((ROOT / "examples/qwen38-27b.json").read_text())["models"][0]
        lever = json.loads((ROOT / "examples/qwen38-27b-k8v4.json").read_text())["models"][0]
        self.assertIs(default.get("long-context"), True)
        self.assertNotIn("kv", default)
        self.assertEqual((lever["kv"], lever["max-context"]), ("k8v4", 175616))


if __name__ == "__main__":
    unittest.main()
