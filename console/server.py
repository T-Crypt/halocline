"""The console's HTTP server: the page, its JSON and live stream, load and unload, and an API pass-through.

    GET  /                       the page
    GET  /api/state              the fleet, the model on the card, its metrics and the host (what the page polls)
    GET  /api/stream             the same as server-sent events, once a second
    GET  /api/history?range=6h   samples averaged into buckets, per-model stats and events
    GET  /api/requests?model=    stored request rows, newest first
    GET  /api/logs?model=&after= engine log lines (the engine's own, then its output)
    GET  /api/models/NAME        one model's setup, its last metrics and its stored stats
    POST /api/models/NAME/load   make it the model on the card (swaps out the console's other engine)
    POST /api/models/NAME/unload drain it and stop it (berth's drain posts here too)
    GET  /v1/models              every model in the fleet, OpenAI-shaped
    POST /v1/...                 passed to the named model's engine, loading it first
"""
from __future__ import annotations

import hmac
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from console.collector import Collector
from console.history import History
from console.supervisor import EXTERNAL, READY, Supervisor

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "console" / "web"
FONTS = ROOT / "serve" / "web" / "fonts"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".svg": "image/svg+xml", ".woff2": "font/woff2", ".png": "image/png",
         ".webmanifest": "application/manifest+json"}
RANGES = {"15m": 900, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
LOAD_WAIT_S = 900                      # how long a passed-through request waits for its model to load


class Console:
    def __init__(self, fleet, api_key: str | None = None, history: History | None = None, telemetry=None):
        self.fleet = fleet
        self.api_key = api_key
        self.history = history if history is not None else History(fleet.data_dir / "history.sqlite3",
                                                                     fleet.retention_days)
        self.sup = Supervisor(fleet, self.history)
        self.col = Collector(self.sup, self.history, telemetry)
        self.started = time.time()

    def state(self, points: int = 120) -> dict:
        snap = self.col.snapshot(points)
        return {"title": self.fleet.title, "now": time.time(), "since": self.started, **self.sup.snapshot(), **snap,
                "holders": self.holders(snap["apps"])}

    def holders(self, apps: list[dict]) -> list[dict]:
        """Each process on the card and whether it is the console's engine (anything else is a squatter here)."""
        ours = {self.sup.runs[n].proc.pid for n in self.sup.runs if self.sup.runs[n].alive()}
        releases = tuple(m.release.rstrip("/") + "/" for m in self.fleet.models)
        out = []
        for a in apps:
            mine = a["pid"] in ours or a.get("path", "").startswith(releases) or _child_of(a["pid"], ours)
            out.append({k: v for k, v in a.items() if k != "path"} | {"ours": mine})
        return out

    def history_view(self, rng: str, model: str | None) -> dict:
        span = RANGES.get(rng, 6 * 3600)
        since = time.time() - span
        return {"range": rng, "series": self.history.series(since, 240, model),
                "stats": self.history.model_stats(since), "stats_all": self.history.model_stats(0),
                "events": self.history.events(since, 300)}


def _child_of(pid: int, parents: set[int]) -> bool:
    """Whether `pid` descends from one of `parents` (berth runs the engine as its child)."""
    seen = 0
    while pid > 1 and seen < 16:
        if pid in parents:
            return True
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        seen += 1
    return False


def make_handler(con: Console):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "halocline-console"

        def log_message(self, fmt, *args):            # quiet: the page polls every second
            pass

        # -------------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-cache"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _authorized(self) -> bool:
            if not con.api_key:
                return True
            got = self.headers.get("Authorization", "")
            token = got[7:] if got.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if not token:
                cookie = self.headers.get("Cookie", "")
                token = next((c.split("=", 1)[1] for c in cookie.split("; ") if c.startswith("hc_key=")), "")
            if hmac.compare_digest(token.encode(), con.api_key.encode()):
                return True
            self._json(401, {"error": {"message": "this console needs its API key"}})
            return False

        def _control_ok(self) -> bool:
            """A control POST comes from this page (same origin) or a non-browser client (berth, curl): a form on another
            site must not load or stop a model."""
            origin = self.headers.get("Origin")
            site = self.headers.get("Sec-Fetch-Site")
            if site and site not in ("same-origin", "none"):
                self._json(403, {"error": {"message": "control requests come from the console's own page"}})
                return False
            if origin:
                host = self.headers.get("Host", "")
                if urlsplit(origin).netloc != host:
                    self._json(403, {"error": {"message": "control requests come from the console's own page"}})
                    return False
            return True

        def _query(self) -> dict:
            return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

        def _static(self, base: Path, name: str):
            f = (base / name).resolve()
            if base.resolve() not in f.parents or f.suffix not in TYPES or not f.is_file():
                self._json(404, {"error": {"message": "not found"}})
                return
            cache = "max-age=86400" if f.suffix in (".woff2", ".png") else "no-cache"
            self._send(200, f.read_bytes(), TYPES[f.suffix], cache)

        # -------------------------------------------------------------- GET
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = unquote(urlsplit(self.path).path).rstrip("/") or "/"
            q = self._query()
            if path == "/":
                return self._static(WEB, "index.html")
            if path in ("/manifest.webmanifest", "/sw.js"):
                return self._static(WEB, path[1:])
            if path.startswith("/web/"):
                return self._static(WEB, path[5:])
            if path.startswith("/fonts/"):
                return self._static(FONTS, path[7:])
            if path == "/health":
                return self._json(200, {"status": "ok", "service": "halocline-console", "active": con.sup.active()})
            if not self._authorized():
                return
            if path == "/api/state":
                return self._json(200, con.state(int(q.get("points", 120))))
            if path == "/api/stream":
                return self._stream()
            if path == "/api/history":
                return self._json(200, con.history_view(q.get("range", "6h"), q.get("model") or None))
            if path == "/api/requests":
                since = time.time() - RANGES.get(q.get("range", "30d"), 30 * 86400)
                return self._json(200, con.history.request_rows(since, q.get("model") or None,
                                                                min(int(q.get("limit", 300)), 2000)))
            if path == "/api/logs":
                name = q.get("model") or con.sup.active()
                if not name or con.fleet.get(name) is None:
                    return self._json(200, {"model": name, "engine": [], "output": []})
                return self._json(200, {"model": name,
                                        "engine": con.col.logs(name, int(q.get("after", 0))),
                                        "output": con.sup.output_since(name, int(q.get("out_after", 0)))})
            if path.startswith("/api/models/"):
                name = path[len("/api/models/"):]
                m = con.fleet.get(name)
                if m is None:
                    return self._json(404, {"error": {"message": f"no model {name!r} in the fleet"}})
                snap = next(x for x in con.sup.snapshot()["models"] if x["name"] == name)
                return self._json(200, {**snap, "last": con.col.model_metrics(name),
                                        "stats": con.history.model_stats(0).get(name)})
            if path in ("/v1/models", "/models"):
                data = [{"id": x["name"], "object": "model", "owned_by": "halocline",
                         "status": {"value": x["state"]}} for x in con.sup.snapshot()["models"]]
                return self._json(200, {"object": "list", "data": data})
            if path.startswith("/v1/"):
                return self._proxy(None)
            self._json(404, {"error": {"message": "not found"}})

        def _stream(self):
            """One state a second. After the first, a key whose value has not changed since this connection last got it
            is left out (and a request list or engine shape the page already holds), so an idle page costs ~2 KB/s."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sent: dict[str, str] = {}
            first = True
            with con.col.lock:
                con.col.viewers += 1
            try:
                while True:
                    st = con.state(300 if first else 1)
                    m = st.get("metrics")
                    if m:
                        st["metrics"] = m = {k: v for k, v in m.items()
                                             if k not in ("history", "hardware_static", "models", "swap")}
                    if st.get("metrics_model") != sent.get("_mm"):       # another engine: send its metrics whole
                        sent = {k: v for k, v in sent.items() if not k.startswith("m.")}
                        sent["_mm"] = st.get("metrics_model")
                    out = {}
                    for k, v in st.items():
                        if k == "metrics" and m:
                            mo = {}
                            for mk, mv in m.items():
                                body = json.dumps(mv, default=str, sort_keys=True)
                                if mk in ("requests", "engine", "best", "totals") and sent.get("m." + mk) == body:
                                    continue
                                sent["m." + mk] = body
                                mo[mk] = mv
                            out[k] = mo
                            continue
                        body = json.dumps(v, default=str, sort_keys=True)
                        if k in ("models", "static", "title", "since") and sent.get(k) == body:
                            continue
                        sent[k] = body
                        out[k] = v
                    self.wfile.write(f"data: {json.dumps(out, default=str)}\n\n".encode())
                    self.wfile.flush()
                    first = False
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                with con.col.lock:
                    con.col.viewers -= 1

        # -------------------------------------------------------------- POST
        def do_POST(self):
            path = unquote(urlsplit(self.path).path).rstrip("/")
            if not self._authorized():
                return
            if path.startswith("/api/models/") and path.rsplit("/", 1)[-1] in ("load", "unload"):
                if not self._control_ok():
                    return
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                name, verb = path[len("/api/models/"):].rsplit("/", 1)
                if con.fleet.get(name) is None:
                    return self._json(404, {"error": {"message": f"no model {name!r} in the fleet"}})
                if verb == "load":
                    # a swap drains the other engine first, which can take minutes: answer now, the page follows it
                    threading.Thread(target=con.sup.load, args=(name,), daemon=True).start()
                    return self._json(202, {"status": "loading", "model": name})
                threading.Thread(target=con.sup.unload, args=(name,), daemon=True).start()
                return self._json(202, {"status": "unloading", "model": name})
            if path.startswith("/v1/") or path in ("/completion", "/tokenize", "/detokenize"):
                n = int(self.headers.get("Content-Length") or 0)
                return self._proxy(self.rfile.read(n) if n else b"")
            self._json(404, {"error": {"message": "not found"}})

        # -------------------------------------------------------------- pass-through
        def _proxy(self, body: bytes | None):
            name = None
            if body:
                try:
                    name = json.loads(body).get("model")
                except (ValueError, AttributeError):
                    name = None
            name = name or con.sup.active() or con.fleet.models[0].name
            m = con.fleet.get(name)
            if m is None:
                return self._json(404, {"error": {"message": f"model {name!r} is not in this Halocline fleet",
                                                  "type": "invalid_request_error"}})
            if con.sup.state[name] not in (READY, EXTERNAL):
                threading.Thread(target=con.sup.load, args=(name,), daemon=True).start()
                end = time.time() + LOAD_WAIT_S
                while time.time() < end and con.sup.state[name] not in (READY, EXTERNAL, "failed"):
                    time.sleep(0.5)
                if con.sup.state[name] not in (READY, EXTERNAL):
                    why = con.sup.errors.get(name) or "it did not load in time"
                    return self._json(503, {"error": {"message": f"{name} is not ready: {why}", "type": "server_error"}})
            conn = http.client.HTTPConnection("127.0.0.1", m.port, timeout=3600)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "connection", "content-length", "accept-encoding")}
            try:
                conn.request(self.command, self.path, body=body if body is not None else None, headers=headers)
                r = conn.getresponse()
            except OSError as e:
                return self._json(502, {"error": {"message": f"{name}: {e}", "type": "server_error"}})
            self.send_response(r.status)
            chunked = r.getheader("Content-Length") is None
            for k, v in r.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length", "server", "date"):
                    self.send_header(k, v)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", r.getheader("Content-Length"))
            self.end_headers()
            try:
                while True:
                    chunk = r.read1(65536) if chunked else r.read(65536)
                    if not chunk:
                        break
                    if chunked:
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    else:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()

    return Handler


def serve(con: Console, host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), make_handler(con))
    httpd.daemon_threads = True
    return httpd
