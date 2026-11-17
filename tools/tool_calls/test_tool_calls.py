"""Tests for tool_calls.py: the case file, the judge, and run + score against a fake OpenAI chat endpoint.

    python3 -m unittest discover -s tools/tool_calls -p 'test_*.py'"""
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tool_calls as T  # noqa: E402

CASES = T.load_cases(Path(T.__file__).with_name("cases.jsonl"))
BY_PROMPT = {c["messages"][0]["content"]: c for c in CASES}


class Fake(BaseHTTPRequestHandler):
    """'oracle' answers every case as labelled; 'always' calls web_search every time."""
    mode = "oracle"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert [t["function"]["name"] for t in body["tools"]] == [t["function"]["name"] for t in T.TOOLS]
        c = BY_PROMPT[body["messages"][0]["content"]]
        msg = {"role": "assistant", "content": "ok"}
        if Fake.mode == "always":
            msg["tool_calls"] = [{"type": "function", "function": {"name": "web_search", "arguments": '{"query": "x"}'}}]
        elif c["expect"] == "call":
            args = {k: "v" for k in c["args"]}
            msg["tool_calls"] = [{"type": "function", "function": {"name": c["tool"], "arguments": json.dumps(args)}}]
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class ToolCallsTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, out, mode):
        Fake.mode = mode
        with redirect_stdout(io.StringIO()):
            self.assertEqual(T.main(["run", "--url", self.url, "--model", "m", "--out", str(self.t / out)]), 0)
        return self.t / out

    def test_the_case_file(self):
        self.assertGreaterEqual(len(CASES), 20)
        self.assertEqual(sum(c["expect"] == "call" for c in CASES), sum(c["expect"] == "none" for c in CASES))
        (self.t / "bad.jsonl").write_text('{"id":"a","expect":"call","tool":"nope","prompt":"x"}\n')
        with self.assertRaises(ValueError):
            T.load_cases(self.t / "bad.jsonl")

    def test_judge(self):
        call = {"expect": "call", "tool": "send_email", "args": ["to", "subject", "body"]}
        none = {"expect": "none"}
        self.assertTrue(T.judge(call, ("send_email", '{"to":"a","subject":"b","body":"c"}'))["right"])
        self.assertFalse(T.judge(call, ("send_email", '{"to":"a"}'))["right"])             # a required key missing
        self.assertFalse(T.judge(call, ("send_email", "{not json"))["right"])
        self.assertFalse(T.judge(call, ("web_search", '{"query":"q"}'))["right"])          # another tool
        self.assertFalse(T.judge(call, None)["right"])
        self.assertTrue(T.judge(none, None)["right"])
        self.assertFalse(T.judge(none, ("calculator", '{"expression":"1"}'))["right"])

    def test_oracle_scores_all_and_agrees_with_itself(self):
        ref = self.run_tool("ref.jsonl", "oracle")
        cand = self.run_tool("cand.jsonl", "oracle")
        s = T.score(cand, ref)
        self.assertEqual(s["all"]["right"], len(CASES))
        self.assertEqual(s["agreement"]["same"], len(CASES))

    def test_always_calling_fails_the_none_cases(self):
        ref = self.run_tool("ref.jsonl", "oracle")
        cand = self.run_tool("cand.jsonl", "always")
        s = T.score(cand, ref)
        self.assertEqual(s["none"]["right"], 0)
        self.assertEqual(s["call"]["right"], sum(c["expect"] == "call" and c["tool"] == "web_search" for c in CASES))
        self.assertIn("none-poem", s["agreement"]["differ"])


if __name__ == "__main__":
    unittest.main()
