"""The console end to end with the mock engine: no GPU, no engine binary, no berth.

Each fleet model runs `python -m serve.server --mock-engine` on its own port, so load, swap, unload, the API pass-through,
the history and the external-engine view run the same code paths they run with real releases.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from console import fleet as fleet_mod
from console.server import Console, serve

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeTelemetry:
    """Host readings without NVML, so the suite runs on any machine."""
    static = {"gpus": [], "cpu_name": "test", "cores": 1, "threads": 1, "driver": None, "cuda": None}

    def snapshot(self):
        return {"now": {"gpus": [{"index": 0, "util": 5, "mem_used": 1 << 30, "mem_total": 24 << 30, "temp": 40,
                                  "power": 50.0}], "cpu": 3.0, "ram_used": 8 << 30, "ram_total": 64 << 30},
                "history": {}, "static": self.static}


def fleet_toml(data: str, port: int, a: int, b: int) -> str:
    def model(name, p):
        return (f'[[model]]\nname = "{name}"\nlabel = "{name} label"\nrelease = "{ROOT}"\nexe = "{sys.executable}"\n'
                f'port = {p}\nargs = ["-m", "serve.server", "--mock-engine"]\n')
    return f'[console]\nport = {port}\ndata = "{data}"\n\n' + model("ModelA", a) + "\n" + model("ModelB", b)


class ConsoleCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.port, cls.pa, cls.pb = free_port(), free_port(), free_port()
        path = Path(cls.tmp.name) / "fleet.toml"
        path.write_text(fleet_toml(cls.tmp.name + "/data", cls.port, cls.pa, cls.pb))
        cls.fleet = fleet_mod.load(str(path))
        cls.con = Console(cls.fleet, telemetry=FakeTelemetry())
        cls.httpd = serve(cls.con, "127.0.0.1", cls.port)
        import threading
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.con.sup.shutdown()
        cls.con.col.stopping.set()
        cls.tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def post(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def state_of(self, name):
        return next(m["state"] for m in self.get("/api/state")["models"] if m["name"] == name)

    def wait(self, cond, what, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.25)
        self.fail(f"timed out waiting for {what}: {self.get('/api/state')['models']}")

    # ------------------------------------------------------------------ the tests run in name order
    def test_1_page_and_state(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn(b"Halocline", r.read())
        for asset in ("/web/app.js", "/web/charts.js", "/web/app.css", "/web/mark.svg", "/fonts/geist-latin-wght.woff2",
                      "/manifest.webmanifest", "/sw.js", "/web/icons/icon-512.png", "/web/icons/icon-maskable-512.png",
                      "/web/icons/apple-touch-icon.png"):
            with urllib.request.urlopen(self.base + asset, timeout=5) as r:
                self.assertEqual(r.status, 200, asset)
        man = self.get("/manifest.webmanifest")
        self.assertEqual(man["display"], "standalone")
        self.assertTrue(any(i["purpose"] == "maskable" and i["sizes"] == "512x512" for i in man["icons"]))
        st = self.get("/api/state")
        self.assertEqual([m["name"] for m in st["models"]], ["ModelA", "ModelB"])
        self.assertIsNone(st["active"])
        self.assertEqual({m["state"] for m in st["models"]}, {"unloaded"})

    def test_2_static_paths_stay_inside(self):
        for bad in ("/web/../server.py", "/web/..%2Fserver.py", "/fonts/../../console/server.py", "/web/nope.js"):
            with self.assertRaises(urllib.error.HTTPError, msg=bad) as cm:
                urllib.request.urlopen(self.base + bad, timeout=5)
            self.assertEqual(cm.exception.code, 404)

    def test_3_control_refuses_other_sites(self):
        code, _ = self.post("/api/models/ModelA/load", headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, _ = self.post("/api/models/ModelA/load", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(code, 403)
        self.assertEqual(self.state_of("ModelA"), "unloaded")
        code, _ = self.post("/api/models/Nope/load")
        self.assertEqual(code, 404)

    def test_4_load_then_swap_through_the_api(self):
        code, body = self.post("/api/models/ModelA/load", headers={"Origin": self.base})
        self.assertEqual(code, 202, body)
        self.wait(lambda: self.state_of("ModelA") == "ready", "ModelA ready")
        self.assertEqual(self.get("/api/state")["active"], "ModelA")
        self.wait(lambda: self.get("/api/state")["metrics"] is not None, "ModelA metrics")
        # a request naming ModelB swaps: A drains and stops, B loads, the request is answered by B
        code, reply = self.post("/v1/chat/completions", {"model": "ModelB", "max_tokens": 8,
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, reply)
        self.assertEqual(self.state_of("ModelA"), "unloaded")
        self.assertEqual(self.state_of("ModelB"), "ready")
        models = {m["id"]: m["status"]["value"] for m in self.get("/v1/models")["data"]}
        self.assertEqual(models, {"ModelA": "unloaded", "ModelB": "ready"})
        # the collector stores B's request; the history keeps the start, ready and stop events
        self.wait(lambda: any(r["model"] == "ModelB" for r in self.get("/api/requests")), "ModelB request stored")
        kinds = [(e["model"], e["kind"]) for e in self.get("/api/history?range=15m")["events"]]
        for want in (("ModelA", "start"), ("ModelA", "ready"), ("ModelA", "stopped"), ("ModelB", "ready")):
            self.assertIn(want, kinds)
        stats = self.get("/api/history?range=15m")["stats_all"]
        self.assertEqual(stats["ModelB"]["requests"], 1)
        self.assertEqual(stats["ModelA"]["load_count"], 1)
        self.assertEqual(self.get("/api/state")["swaps"], 1)

    def test_5_logs_and_model_view(self):
        logs = self.get("/api/logs?model=ModelB")
        self.assertTrue(logs["engine"] or logs["output"])
        one = self.get("/api/models/ModelB")
        self.assertEqual(one["state"], "ready")
        self.assertEqual(one["stats"]["requests"], 1)

    def test_5b_stream_sends_deltas(self):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/api/stream")
        r = c.getresponse()
        msgs = []
        while len(msgs) < 3:
            line = r.fp.readline()
            if line.startswith(b"data: "):
                msgs.append(json.loads(line[6:]))
        c.close()
        self.assertIn("models", msgs[0])
        self.assertNotIn("models", msgs[2])              # unchanged since the first message
        self.assertIn("live", msgs[2])

    def test_6_unload(self):
        code, _ = self.post("/api/models/ModelB/unload")          # berth's drain posts with no Origin
        self.assertEqual(code, 202)
        self.wait(lambda: self.state_of("ModelB") == "unloaded", "ModelB unloaded")
        self.assertIsNone(self.get("/api/state")["active"])

    def test_7_an_engine_started_elsewhere_is_watched_not_controlled(self):
        p = subprocess.Popen([sys.executable, "-m", "serve.server", "--mock-engine", "--host", "127.0.0.1",
                              "--port", str(self.pa), "--alias", "ModelA"], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.wait(lambda: self.state_of("ModelA") == "external", "ModelA external")
            self.assertEqual(self.get("/api/state")["active"], "ModelA")
            self.post("/api/models/ModelA/unload")
            time.sleep(2)
            self.assertIsNone(p.poll(), "the console stopped an engine it did not start")
        finally:
            p.terminate()
            p.wait(10)
        self.wait(lambda: self.state_of("ModelA") == "unloaded", "ModelA gone")


class FleetFileCase(unittest.TestCase):
    def parse(self, text):
        import tomllib
        return fleet_mod.parse(tomllib.loads(text), Path("/srv"))

    def test_good(self):
        f = self.parse('[console]\nport = 9000\nberth = "berth"\n[[model]]\nname = "A"\nrelease = "rel"\nport = 8001\n'
                       'args = ["serve", "a.halo"]\nlease = "a"\nmin_turn = "5m"\n')
        m = f.models[0]
        self.assertEqual(m.release, "/srv/rel")
        self.assertEqual(m.engine_argv(), ["/srv/rel/bin/halocline", "serve", "a.halo", "--host", "127.0.0.1",
                                           "--port", "8001", "--alias", "A"])
        from console.supervisor import Supervisor
        sup = Supervisor.__new__(Supervisor)
        sup.fleet = f
        argv = Supervisor.argv(sup, m)
        self.assertEqual(argv[:4], ["berth", "run", "--as", "a"])
        self.assertIn("http://127.0.0.1:9000/api/models/A/unload", argv)
        self.assertEqual(argv[argv.index("--") + 1], "/srv/rel/bin/halocline")

    def test_refused(self):
        base = '[[model]]\nname = "A"\nrelease = "r"\nport = 8001\nargs = []\n'
        for text, word in ((base + base, "named twice"),
                           (base + base.replace('"A"', '"B"'), "port 8001"),
                           ('[[model]]\nname = "A"\nport = 1\nargs = []\n', "release"),
                           (base.replace("args = []", 'args = "x"'), "list of strings"),
                           (base + "colour = 1\n", "not a model setting"),
                           ("[console]\nport = 8001\n" + base, "engine port"),
                           ("[console]\nwat = 1\n" + base, "not a console setting"),
                           ("[console]\n", "at least one")):
            with self.assertRaises(ValueError, msg=text) as cm:
                self.parse(text)
            self.assertIn(word, str(cm.exception))


class HoldersCase(unittest.TestCase):
    def test_release_paths_are_ours(self):
        f = fleet_mod.parse({"model": [{"name": "A", "release": "/opt/rel/abc", "port": 8001, "args": []}]}, Path("/"))
        con = Console.__new__(Console)
        con.fleet = f
        con.sup = type("S", (), {"runs": {}})()
        rows = con.holders([{"pid": 999999, "name": "halocline", "path": "/opt/rel/abc/bin/halocline", "mib": 20000},
                            {"pid": 999998, "name": "node", "path": "/usr/bin/node", "mib": 2000}])
        self.assertEqual([r["ours"] for r in rows], [True, False])
        self.assertNotIn("path", rows[0])


if __name__ == "__main__":
    unittest.main()


class GatewayModeCase(unittest.TestCase):
    """Berth's gateway starts and stops the engines; the console asks it to and passes requests through it."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        cls.tmp = tempfile.TemporaryDirectory()
        cls.port, cls.pe, cls.gport = free_port(), free_port(), free_port()
        keyfile = Path(cls.tmp.name) / "api.key"
        keyfile.write_text("sekrit\n")
        cls.calls, cls.procs = [], {}
        case = cls

        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                case.calls.append((self.path, body, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != "Bearer sekrit":
                    self.send_response(401); self.end_headers(); return
                if self.path == "/v1/berth/switch":
                    case.procs["e"] = subprocess.Popen(
                        [sys.executable, "-m", "serve.server", "--mock-engine", "--host", "127.0.0.1", "--port",
                         str(case.pe), "--alias", body["model"]], cwd=ROOT, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    out = b'{"status":"switched"}'
                elif self.path == "/v1/berth/stop":
                    p = case.procs.pop("e", None)
                    if p:
                        p.terminate(); p.wait(10)
                    out = b'{"status":"stopped"}'
                else:                                 # a chat request: answered as the gateway would, by the engine
                    req = urllib.request.Request(f"http://127.0.0.1:{case.pe}{self.path}", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    out = urllib.request.urlopen(req, timeout=60).read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        cls.gw = ThreadingHTTPServer(("127.0.0.1", cls.gport), Gateway)
        threading.Thread(target=cls.gw.serve_forever, daemon=True).start()
        path = Path(cls.tmp.name) / "fleet.toml"
        path.write_text(f'[console]\nport = {cls.port}\ndata = "{cls.tmp.name}/data"\n'
                        f'gateway = "http://127.0.0.1:{cls.gport}"\ngateway_key_file = "{keyfile}"\n\n'
                        f'[[model]]\nname = "Sparse"\nrelease = "{ROOT}"\nport = {cls.pe}\nargs = []\n'
                        f'lease = "halocline"\naliases = ["router-halocline"]\n')
        cls.con = Console(fleet_mod.load(str(path)), telemetry=FakeTelemetry())
        cls.httpd = serve(cls.con, "127.0.0.1", cls.port)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs.values():
            p.terminate(); p.wait(10)
        cls.httpd.shutdown(); cls.gw.shutdown()
        cls.con.sup.stopping.set(); cls.con.col.stopping.set()
        cls.tmp.cleanup()

    def state(self):
        return json.load(urllib.request.urlopen(self.base + "/api/state", timeout=10))["models"][0]

    def wait(self, want, timeout=60):
        end = time.time() + timeout
        while time.time() < end and self.state()["state"] != want:
            time.sleep(0.25)
        self.assertEqual(self.state()["state"], want)

    def test_load_alias_request_and_stop(self):
        req = urllib.request.Request(self.base + "/api/models/Sparse/load", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        self.assertEqual(urllib.request.urlopen(req, timeout=10).status, 202)
        self.wait("ready")
        self.assertEqual(self.calls[0], ("/v1/berth/switch", {"engine": "halocline", "model": "Sparse"}, "Bearer sekrit"))
        # an alias reaches the engine renamed (it answers only to its own name); berth saw only the switch
        req = urllib.request.Request(self.base + "/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "router-halocline", "max_tokens": 4,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer client"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.assertEqual(json.loads(r.read())["model"], "Sparse")
        # and the Anthropic API, which berth's gateway does not route, works the same way
        req = urllib.request.Request(self.base + "/v1/messages", method="POST",
                                     data=json.dumps({"model": "router-halocline", "max_tokens": 4,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.assertEqual(r.status, 200)
        self.assertEqual([c[0] for c in self.calls], ["/v1/berth/switch"])
        ids = {m["id"] for m in json.load(urllib.request.urlopen(self.base + "/v1/models", timeout=10))["data"]}
        self.assertEqual(ids, {"Sparse", "router-halocline"})
        req = urllib.request.Request(self.base + "/api/models/Sparse/unload", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        self.wait("unloaded")
        self.assertEqual(self.calls[-1][:2], ("/v1/berth/stop", {"engine": "halocline"}))

    def test_unloaded_request_asks_berth_first(self):
        n = len(self.calls)
        req = urllib.request.Request(self.base + "/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "Sparse", "max_tokens": 4,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        self.assertEqual(urllib.request.urlopen(req, timeout=90).status, 200)
        self.assertEqual(self.calls[n][0], "/v1/berth/switch")
