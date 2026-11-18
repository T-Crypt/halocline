"""The build's checks, each one shown failing on a planted input and passing clean."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))

import build  # noqa: E402
import extract  # noqa: E402


class TemplateNumberTests(unittest.TestCase):
    def test_clean_template_has_no_literal_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ok.html").write_text("<p>mean KLD {{ v('sparse.quality.mean_kld') }}</p>")
            self.assertEqual(build.template_literals(Path(tmp)), [])

    def test_planted_number_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.html").write_text("<p>mean KLD 0.08195</p>")
            problems = build.template_literals(Path(tmp))
            self.assertEqual(len(problems), 1)
            self.assertIn("literal number", problems[0])

    def test_planted_number_in_alt_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.html").write_text('<img src="x.svg" alt="needle at 36K">')
            problems = build.template_literals(Path(tmp))
            self.assertEqual(len(problems), 1)
            self.assertIn("attribute", problems[0])

    def test_real_templates_are_clean(self):
        self.assertEqual(build.template_literals(ROOT / "site" / "templates"), [])


class ContentNumberTests(unittest.TestCase):
    def test_clean_claim(self):
        findings = [{"id": "x", "title": "A claim", "claim": "KLD {kld}", "caveat": "Measured."}]
        self.assertEqual(build.content_problems(findings, [], []), [])

    def test_planted_number_in_a_claim_fails(self):
        findings = [{"id": "x", "title": "A claim", "claim": "KLD 0.08", "caveat": "Measured."}]
        problems = build.content_problems(findings, [], [])
        self.assertEqual(len(problems), 1)
        self.assertIn("literal digit", problems[0])

    def test_planted_number_in_an_open_item_fails(self):
        items = [{"title": "Full context", "status": "open", "accept": "at 262K"}]
        problems = build.content_problems([], [], items)
        self.assertEqual(len(problems), 1)

    def test_planted_number_in_a_rounding_fails(self):
        rows = [{"part": "dense", "rounding": "Cache", "when": "every request", "cost": "costs 0.00093"}]
        problems = build.content_problems([], rows, [])
        self.assertEqual(len(problems), 1)

    def test_real_content_is_clean(self):
        self.assertEqual(build.content_literals(), [])


class MissingKeyTests(unittest.TestCase):
    def test_resolver_names_the_missing_key(self):
        v = build.resolver({})
        with self.assertRaises(build.BuildError) as caught:
            v("sparse.quality.mean_kld")
        self.assertIn("sparse.quality.mean_kld", str(caught.exception))

    def test_resolver_returns_a_value(self):
        data = build.load_data()
        v = build.resolver(data)
        self.assertAlmostEqual(v("sparse.quality.mean_kld").raw, 0.0819529, places=7)
        self.assertEqual(str(v("sparse.kv.arms.int8.label")), "8-bit")


class OriginTests(unittest.TestCase):
    def test_allowed_origins_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ok.html").write_text(
                '<a href="https://github.com/T-Crypt/halocline/blob/main/README.md">repo</a>'
                '<a href="b.html">local</a>'
                '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
            )
            self.assertEqual(build.check_origins(Path(tmp)), [])

    def test_planted_third_party_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.html").write_text('<script src="https://cdn.example.com/x.js"></script>')
            problems = build.check_origins(Path(tmp))
            self.assertEqual(len(problems), 1)
            self.assertIn("cdn.example.com", problems[0])

    def test_planted_protocol_relative_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.html").write_text('<link rel="stylesheet" href="//fonts.example.com/x.css">')
            problems = build.check_origins(Path(tmp))
            self.assertEqual(len(problems), 1)


class LinkTests(unittest.TestCase):
    def test_local_links_and_anchors_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.html").write_text('<div id="top"></div><a href="b.html">b</a><a href="b.html#ok">b</a><a href="#top">top</a>')
            Path(tmp, "b.html").write_text('<div id="ok"></div>')
            self.assertEqual(build.check_links(Path(tmp)), [])

    def test_planted_broken_links_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.html").write_text('<a href="missing.html">missing</a><a href="#nope">anchor</a>')
            problems = build.check_links(Path(tmp))
            self.assertEqual(len(problems), 2)

    def test_real_output_links_resolve(self):
        out = ROOT / "site" / "_out"
        if not out.exists():
            self.skipTest("run site/build.py first")
        self.assertEqual(build.check_links(out), [])


class WeightTests(unittest.TestCase):
    def test_planted_heavy_page_fails_the_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "big.css").write_bytes(b"x" * (401 * 1024))
            Path(tmp, "page.html").write_text('<link rel="stylesheet" href="big.css">')
            weights = dict(build.check_weight(Path(tmp)))
            self.assertGreater(weights["page.html"], 400 * 1024)

    def test_real_pages_are_under_budget(self):
        out = ROOT / "site" / "_out"
        if not out.exists():
            self.skipTest("run site/build.py first")
        for name, size in build.check_weight(out):
            self.assertLess(size, 400 * 1024, name)


class SourceTests(unittest.TestCase):
    def test_each_source_form_maps_to_a_link(self):
        self.assertTrue(build.source_href("evals/2026-11-17/halocline-kv-kld/int8/bench.json#quality.kld_mean").endswith("/tree/main/evals/2026-11-17/halocline-kv-kld/int8"))
        self.assertTrue(build.source_href("README.md line 12").endswith("/blob/main/README.md#L12"))
        self.assertIsNone(build.source_href("computed: the two ABBA arms"))
        self.assertIsNone(build.source_href("plan: SITE_PLAN.md section 2"))

    def test_committed_sources_exist(self):
        self.assertEqual(build.source_problems(), [])

    def test_every_content_reference_resolves(self):
        data = build.load_data()
        v = build.resolver(data)
        findings, roundings, open_items = build.load_content(v)
        self.assertGreaterEqual(len(findings), 10)
        self.assertGreaterEqual(len(roundings), 10)
        self.assertGreaterEqual(len(open_items), 5)
        for item in findings:
            self.assertTrue(item["claim"])
            self.assertTrue(item["eval_links"] or item["doc_links"])


class FormatTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(extract.format_value(0.0819529, "kld"), "0.08195")
        self.assertEqual(extract.format_value(90.4549, "pct"), "90.45%")
        self.assertEqual(extract.format_value(6052.56, "tps"), "6,053")
        self.assertEqual(extract.format_value(2459, "ms"), "2,459 ms")
        self.assertEqual(extract.format_value(0.00023, "pm"), "\u00b1 0.00023")
        self.assertEqual(extract.format_value([0.00074, 0.00113], "kld_ci"), "[+0.00074, +0.00113]")

    def test_escape(self):
        self.assertEqual(build.esc('<script>alert("x")</script>'), "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;")


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = build.load_data()
        cls.charts = build.build_charts(build.resolver(data))

    def test_every_chart_has_a_title_desc_and_table(self):
        for name, chart in self.charts.items():
            with self.subTest(chart=name):
                self.assertIn('role="img"', chart.svg)
                self.assertIn("<title", chart.svg)
                self.assertIn("<desc", chart.svg)
                self.assertTrue(chart.headers)
                self.assertTrue(chart.rows)
                self.assertGreater(len(chart.svg), 500)

    def test_every_chart_names_its_runs(self):
        for name, chart in self.charts.items():
            with self.subTest(chart=name):
                self.assertTrue(chart.sources)

    def test_rendered_charts_have_a_table_fallback(self):
        page = ROOT / "site" / "_out" / "index.html"
        if not page.exists():
            self.skipTest("run site/build.py first")
        out = page.read_text()
        self.assertEqual(out.count('class="chart"'), out.count('class="table-view"'))
        self.assertEqual(out.count('class="chart-svg"'), out.count('class="table-view"'))


if __name__ == "__main__":
    unittest.main()
