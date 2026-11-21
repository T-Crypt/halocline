"""The extractors read the repository; these tests pin the keys they read."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))

import extract  # noqa: E402


class BenchTests(unittest.TestCase):
    def test_sparse_kv_quality(self):
        b = extract.bench("evals/2026-11-17/halocline-kv-kld/int8/bench.json")
        self.assertEqual(b["quality"]["chunks"], 100)
        self.assertEqual(b["quality"]["rows"], 25500)
        self.assertAlmostEqual(b["quality"]["kld_mean"], 0.0819529, places=7)
        self.assertAlmostEqual(b["quality"]["top1_pct"], 90.4549, places=4)
        self.assertAlmostEqual(b["summary"]["pp131072"]["tok_s_mean"], 6019.34, places=2)

    def test_dense_kv_quality(self):
        b = extract.bench("evals/2026-11-17/halocline-d8-kv/k8v4/bench.json")
        self.assertAlmostEqual(b["quality"]["kld_mean"], 0.00675654, places=8)
        self.assertEqual(b["commit"], "07f11e6734")

    def test_paired_intervals(self):
        p = extract.parse_paired("evals/2026-11-17/halocline-kv-kld/paired.md")
        self.assertAlmostEqual(p["fp16"]["delta"], 0.00035, places=5)
        self.assertAlmostEqual(p["fp16"]["lo"], -0.00061, places=5)
        self.assertAlmostEqual(p["fp16"]["hi"], 0.00131, places=5)

    def test_paired_vs_k8v4_is_kept_apart(self):
        p = extract.parse_paired("evals/2026-11-17/halocline-d8-kv/paired.md")
        self.assertAlmostEqual(p["q4"]["delta"], 0.00197, places=5)
        self.assertAlmostEqual(p["q4@vs_k8v4"]["delta"], 0.00104, places=5)

    def test_kld_text(self):
        k = extract.parse_kld_txt("evals/2026-11-17/halocline-dense-quality/llama-q4kxl-f16/kld.txt")
        self.assertAlmostEqual(k["mean_kld"], 0.005730, places=6)
        self.assertAlmostEqual(k["mean_kld_err"], 0.000230, places=6)
        self.assertAlmostEqual(k["same_top"], 96.541, places=3)

    def test_smoke_details(self):
        s = extract.parse_smoke("evals/2026-11-17/halocline-llama-swap/smoke.json")
        self.assertAlmostEqual(s["warm_low"], 147.8)
        self.assertAlmostEqual(s["warm_high"], 181.6)
        self.assertEqual(s["needle"]["decode"], 123.1)
        self.assertEqual(s["needle"]["prompt"], 6298)
        self.assertEqual(s["needle"]["tokens"], 36467)

    def test_dense_fit_probe(self):
        data = extract.build_dense().as_json()["values"]
        self.assertEqual(data["fit"]["q4kxl"]["need_mib"], 11830)
        self.assertEqual(data["fit"]["q4kxl"]["free_mib"], 7597)
        self.assertEqual(data["fit"]["iq4xs"]["free_mib"], 10589)

    def test_rows_exhibit(self):
        ex, rel = extract.rows_exhibit("evals/2026-11-17/halocline-kv-kld/rows/int8.tsv", compressed=False, label="sparse")
        self.assertEqual(rel, "evals/2026-11-17/halocline-kv-kld/rows/int8.tsv")
        self.assertEqual(ex["first_position"], 256)
        self.assertEqual(ex["last_position"], 510)
        self.assertEqual(len(ex["base_p"]), ex["n_positions"])
        self.assertEqual(len(ex["test_p"]), ex["n_positions"])
        self.assertEqual(len(ex["disagree_flags"]), ex["n_positions"])
        self.assertTrue(0.0 < ex["mean_kld"] < 1.0)

    def test_rows_exhibit_compressed(self):
        ex, _ = extract.rows_exhibit(
            "evals/2026-11-17/halocline-dense-quality/q4kxl-int8-mmvq/rows.tsv.xz", compressed=True, label="dense"
        )
        self.assertEqual(len(ex["positions"]), ex["n_positions"])
        self.assertGreater(ex["max_kld"], 0.0)


class DataFileTests(unittest.TestCase):
    def test_committed_data_is_current(self):
        fresh = extract.build_all()
        for name, data in fresh.items():
            path = ROOT / "site" / "data" / f"{name}.json"
            self.assertTrue(path.exists(), path)
            self.assertEqual(json.loads(path.read_text()), data, f"{name}.json is stale")

    def test_every_leaf_has_a_source(self):
        for name, data in extract.build_all().items():
            values = data["values"]
            sources = data["_sources"]
            formats = data["_formats"]

            def walk(node, prefix=""):
                if isinstance(node, dict):
                    for key, child in node.items():
                        walk(child, f"{prefix}.{key}" if prefix else key)
                    return
                if prefix.endswith((".base_p", ".test_p", ".disagree_flags")):
                    return
                self.assertIn(prefix, sources, f"{name}: {prefix} has no source")
                self.assertIn(prefix, formats, f"{name}: {prefix} has no format")

            walk(values)

    def test_every_line_source_states_its_value(self):
        # A source "FILE line N" must still say the value: on that line, or further down the paragraph, table or code
        # block it opens. Every word of a text value counts (a label can be assembled from one line's parts); a number
        # may be written with or without thousands separators. This catches a pointer left behind when the file moved
        # (2026-11-21: README line 33 and lines 329-333 had drifted onto other text).
        def forms(value):
            if isinstance(value, bool):
                return None
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, int):
                return [[str(value)], [f"{value:,}"]]
            if isinstance(value, float):
                return [[f"{value:.{k}f}"] for k in range(0, 7)] + [[repr(value)]]
            if isinstance(value, str) and value.strip():
                return [value.replace("`", "").replace("*", "").split()]
            return None

        def block(lines, n):
            first = lines[n - 1]
            if first.lstrip().startswith("```"):
                end = next((i for i in range(n, len(lines)) if lines[i].lstrip().startswith("```")), len(lines))
                return lines[n - 1 : end + 1]
            if first.lstrip().startswith("|"):
                return [first]
            end = next((i for i in range(n, len(lines)) if not lines[i].strip()), len(lines))
            return lines[n - 1 : end]

        checked = 0
        for name, data in extract.build_all().items():
            for path, source in data["_sources"].items():
                m = re.fullmatch(r"(.+) line (\d+)", source)
                if not m:
                    continue
                rel, n = m.group(1), int(m.group(2))
                node = data["values"]
                for part in path.split("."):
                    node = node[part]
                options = forms(node)
                if options is None:
                    continue
                lines = (ROOT / rel).read_text(errors="replace").splitlines()
                self.assertLessEqual(n, len(lines), f"{name}: {path}: {source} is past the end of the file")
                text = " ".join(block(lines, n)).replace("`", "").replace("*", "")
                said = any(all(word in text for word in words) for words in options)
                self.assertTrue(said or (path.endswith(".line") and node == n),
                                f"{name}: {path} = {node!r}, but {source} reads {lines[n - 1][:80]!r}")
                checked += 1
        self.assertGreater(checked, 150)

    def test_sparse_architecture_facts(self):
        data = extract.build_sparse().as_json()["values"]
        self.assertEqual(data["facts"]["layers"], 48)
        self.assertEqual(data["facts"]["experts"], 512)
        self.assertEqual(data["facts"]["active_routed"], 10)
        self.assertEqual(data["facts"]["native_context"], 262144)

    def test_dense_depth_grid(self):
        data = extract.build_dense().as_json()["values"]
        self.assertEqual(data["depth16"]["tokens"], 16384)
        self.assertEqual(data["depth36"]["tokens"], 36864)


if __name__ == "__main__":
    unittest.main()
