"""Tests for sampling_dist.py against a fake OpenAI chat endpoint that samples from a fixed distribution per prompt.

    python3 -m unittest discover -s tools/sampling_dist -p 'test_*.py'"""
import io
import json
import math
import random
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling_dist as S  # noqa: E402

TOKENS = [f"t{i}" for i in range(12)]


class Fake(BaseHTTPRequestHandler):
    """A model whose every row is the same softmax over 12 tokens, shifted by the prompt; `mode` picks the sampler:
    'fair' samples the distribution with the request's seed, 'greedy' always takes the top token."""
    mode = "fair"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        shift = sum(map(ord, body["messages"][0]["content"])) % 5
        w = [math.exp(-0.45 * ((i + shift) % len(TOKENS))) for i in range(len(TOKENS))]
        z = sum(w)
        lp = [math.log(x / z) for x in w]
        order = sorted(range(len(TOKENS)), key=lambda i: -lp[i])
        rng = random.Random(body["seed"] * 7919 + shift)
        content = []
        for _ in range(body["max_tokens"]):
            k = order[0] if self.mode == "greedy" else rng.choices(range(len(TOKENS)), weights=w)[0]
            content.append({"token": TOKENS[k], "logprob": lp[k],
                            "top_logprobs": [{"token": TOKENS[j], "logprob": lp[j]} for j in order[:body["top_logprobs"]]]})
        out = json.dumps({"choices": [{"logprobs": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class SamplingDistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        prompts = [{"id": f"p{i}", "prompt": f"question number {i} " + "x" * i} for i in range(12)]
        (self.t / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
        Fake.mode = "fair"

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, seeds, out, mode="fair"):
        Fake.mode = mode
        with redirect_stdout(io.StringIO()):
            rc = S.main(["run", "--url", self.url, "--model", "m", "--prompts", str(self.t / "prompts.jsonl"),
                         "--seeds", seeds, "--max-tokens", "16", "--out", str(self.t / out)])
        self.assertEqual(rc, 0)
        return self.t / out

    def test_seeds_and_prompts(self):
        self.assertEqual(S.seeds_of("0-3,7"), [0, 1, 2, 3, 7])
        (self.t / "dup.jsonl").write_text('{"id":"a","prompt":"x"}\n{"id":"a","prompt":"y"}\n')
        with self.assertRaises(ValueError):
            S.load_prompts(self.t / "dup.jsonl")

    def test_run_records_every_prompt_and_seed(self):
        p = self.run_tool("0-3", "a.jsonl")
        meta, by = S.load_run(p)
        self.assertEqual(meta["seeds"], [0, 1, 2, 3])
        self.assertEqual(len(by), 12)
        self.assertTrue(all(len(r["picks"]) == 16 for recs in by.values() for r in recs))

    def test_the_same_build_passes_and_matches_exactly(self):
        a, b, c = self.run_tool("0-15", "a.jsonl"), self.run_tool("16-31", "b.jsonl"), self.run_tool("0-15", "c.jsonl")
        rows = S.compare(a, b, c)
        self.assertTrue(rows["pass"], rows)
        for k in S.METRICS:
            self.assertEqual(rows[k]["cand_minus_a"], 0.0)   # the candidate shares A's seeds and sampler
        self.assertEqual(rows["first_tv"]["cand_vs_a"], 0.0)
        top1 = rows["top1"]["a"]
        self.assertTrue(0.2 < top1 < 0.6, top1)               # the fake's top token holds about a third of the mass

    def test_a_greedy_sampler_fails(self):
        a, b = self.run_tool("0-15", "a.jsonl"), self.run_tool("16-31", "b.jsonl")
        c = self.run_tool("0-15", "c.jsonl", mode="greedy")
        rows = S.compare(a, b, c)
        self.assertFalse(rows["pass"])
        self.assertFalse(rows["top1"]["pass"])
        self.assertGreater(rows["top1"]["cand_minus_a"], 0.3)

    def test_compare_cli(self):
        a, b, c = self.run_tool("0-7", "a.jsonl"), self.run_tool("8-15", "b.jsonl"), self.run_tool("0-7", "c.jsonl")
        with redirect_stdout(io.StringIO()) as out:
            rc = S.main(["compare", str(a), str(b), str(c)])
        self.assertEqual(rc, 0)
        self.assertIn("PASS", out.getvalue())


if __name__ == "__main__":
    unittest.main()
