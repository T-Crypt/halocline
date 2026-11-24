"""The supervisor: starts one model's engine at a time, swaps between them and stops them.

Each model's engine is its own release's `bin/halocline serve`, optionally wrapped in `berth run` so the card is
leased (berth drains whatever else holds it first). A model is in one of these states:

    unloaded   nothing answers on its port
    starting   the console started it; its port does not answer yet
    loading    its /health answers and says it is loading (with a percent)
    ready      its /health says loaded
    stopping   the console is draining or stopping it
    failed     its last start ended before it was ready (the error and the log tail say why)
    external   something the console did not start answers on its port (shown, never stopped)
"""
from __future__ import annotations

import collections
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from console.fleet import Fleet, Model

UNLOADED, STARTING, LOADING, READY, STOPPING, FAILED, EXTERNAL = (
    "unloaded", "starting", "loading", "ready", "stopping", "failed", "external")
LIVE = (STARTING, LOADING, READY, STOPPING)
LOG_KEEP = 4000                       # engine output lines kept in memory per model
LOG_FILE_MAX = 32 * 2**20             # the per-model log file rolls over at this size


def get_json(url: str, timeout: float = 2.0):
    """The URL's JSON, or None when nothing answers or the answer is not JSON."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError, urllib.error.URLError):
        return None


class EngineRun:
    """One start of one model: its process, its output and how the start went."""

    def __init__(self, model: Model, argv: list[str], log_path: Path, on_line):
        self.model, self.argv = model, argv
        self.started = time.time()
        self.ready_at: float | None = None
        self.ended_at: float | None = None
        self.exit_code: int | None = None
        self.percent: float | None = None
        env = {**os.environ, **model.env}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > LOG_FILE_MAX:
            log_path.replace(log_path.with_suffix(".log.1"))
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        self.log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(argv)}\n")
        self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                     cwd=model.release if os.path.isdir(model.release) else None)
        self.on_line = on_line
        threading.Thread(target=self._read, daemon=True, name=f"out-{model.name}").start()

    def _read(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            self.log_file.write(line + "\n")
            self.on_line(self.model.name, line)
        self.exit_code = self.proc.wait()
        self.ended_at = time.time()
        self.log_file.write(f"=== exit {self.exit_code}\n")
        self.log_file.close()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def signal(self, sig) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class Supervisor:
    def __init__(self, fleet: Fleet, history=None, poll_s: float = 1.0):
        self.fleet = fleet
        self.history = history
        self.lock = threading.RLock()
        self.switch = threading.Lock()                       # one load or swap at a time
        self.runs: dict[str, EngineRun] = {}                 # the console's current (or last) run per model
        self.state: dict[str, str] = {m.name: UNLOADED for m in fleet.models}
        self.health: dict[str, dict | None] = {m.name: None for m in fleet.models}
        self.errors: dict[str, str] = {}
        self.output: dict[str, collections.deque] = {m.name: collections.deque(maxlen=LOG_KEEP) for m in fleet.models}
        self.out_id = 0
        self.swap: dict | None = None                        # {"from", "to", "phase", "since"}
        self.swaps = 0
        self.stopping = threading.Event()
        self.poll_s = poll_s
        threading.Thread(target=self._watch, daemon=True, name="supervisor").start()

    # ------------------------------------------------------------------ the engine's output, for the page's log
    def _line(self, name: str, line: str) -> None:
        with self.lock:
            self.out_id += 1
            self.output[name].append({"id": self.out_id, "t": time.time(), "src": "out", "lvl": _level(line),
                                      "msg": line})

    def output_since(self, name: str, after: int, limit: int = 500) -> list[dict]:
        with self.lock:
            rows = [r for r in self.output.get(name, ()) if r["id"] > after]
        return rows[-limit:]

    # ------------------------------------------------------------------ state
    def active(self) -> str | None:
        """The model on the card: the console's live run, else one that answers on its port."""
        with self.lock:
            for want in (READY, LOADING, STARTING, STOPPING, EXTERNAL):
                for m in self.fleet.models:
                    if self.state[m.name] == want:
                        return m.name
        return None

    def snapshot(self) -> dict:
        with self.lock:
            out = []
            for m in self.fleet.models:
                run = self.runs.get(m.name)
                h = self.health.get(m.name) or {}
                out.append({**m.public(), "state": self.state[m.name], "error": self.errors.get(m.name),
                            "percent": h.get("percent") if self.state[m.name] == LOADING else None,
                            "pid": run.proc.pid if run and run.alive() else None,
                            "started": run.started if run else None,
                            "ready_at": run.ready_at if run else None,
                            "load_s": round(run.ready_at - run.started, 1) if run and run.ready_at else None,
                            "exit_code": run.exit_code if run else None,
                            "max_context": h.get("max_context"), "images": h.get("images")})
            return {"models": out, "active": self.active(), "swap": dict(self.swap) if self.swap else None,
                    "swaps": self.swaps}

    def _event(self, model: str, kind: str, detail: dict | None = None) -> None:
        if self.history is not None:
            self.history.event(model, kind, detail or {})

    # ------------------------------------------------------------------ watching
    def _watch(self):
        while not self.stopping.is_set():
            for m in self.fleet.models:
                try:
                    self._check(m)
                except Exception as e:  # noqa: BLE001 - the watcher must keep running
                    self._line(m.name, f"console: health check failed: {e}")
            self.stopping.wait(self.poll_s)

    def _check(self, m: Model) -> None:
        h = get_json(m.base + "/health", timeout=1.5)
        with self.lock:
            self.health[m.name] = h
            run = self.runs.get(m.name)
            ours = run is not None and run.alive()
            state = self.state[m.name]
            if ours:
                if state == STOPPING:
                    return
                if h is None:
                    self.state[m.name] = STARTING
                elif h.get("loading") or not h.get("loaded", True):
                    self.state[m.name] = LOADING
                    run.percent = h.get("percent")
                else:
                    if state != READY:
                        run.ready_at = time.time()
                        self.errors.pop(m.name, None)
                        self._event(m.name, "ready", {"load_s": round(run.ready_at - run.started, 1)})
                    self.state[m.name] = READY
                return
            if run is not None and state in LIVE and state != STOPPING:
                # our run ended on its own: a failed start, a crash, or berth drained it for another engine
                tail = [r["msg"] for r in list(self.output[m.name])[-12:]]
                why = f"exited with code {run.exit_code}" if run.ready_at is None else "stopped"
                if run.ready_at is None:
                    self.state[m.name] = FAILED
                    self.errors[m.name] = why
                    self._event(m.name, "failed", {"exit": run.exit_code, "tail": tail})
                else:
                    self.state[m.name] = UNLOADED
                    self._event(m.name, "exited", {"exit": run.exit_code})
                return
            if state == STOPPING:
                return
            if h is not None and h.get("status") == "ok":
                if state != EXTERNAL:
                    self._event(m.name, "external", {})
                self.state[m.name] = EXTERNAL
            elif state == EXTERNAL:
                self.state[m.name] = UNLOADED
                self._event(m.name, "external-gone", {})

    # ------------------------------------------------------------------ control
    def argv(self, m: Model) -> list[str]:
        engine = m.engine_argv()
        if not (self.fleet.berth and m.lease):
            return engine
        own = f"http://{_loopback(self.fleet.host)}:{self.fleet.port}"
        argv = [self.fleet.berth, "run", "--as", m.lease, "--priority", m.priority]
        if self.fleet.berth_config:
            argv += ["--config", self.fleet.berth_config]
        if m.min_turn:
            argv += ["--min-turn", m.min_turn]
        # berth asks the console to unload, so a drain goes through the same idle wait a click does
        argv += ["--drain-post", f"{own}/api/models/{m.name}/unload", "--drain-slots", m.base, "--"]
        return argv + engine

    def load(self, name: str) -> dict:
        """Make `name` the model on the card: stop the console's other engine first. Returns once it is started;
        the watcher follows it to ready."""
        m = self.fleet.get(name)
        if m is None:
            raise KeyError(name)
        with self.switch:
            with self.lock:
                if self.state[name] in (READY, LOADING, STARTING):
                    return {"status": self.state[name]}
                if self.state[name] == EXTERNAL:
                    return {"status": "external"}
                current = next((o for o in self.fleet.models if o.name != name and self.state[o.name] in LIVE), None)
                self.swap = {"from": current.name if current else None, "to": name,
                             "phase": "draining" if current else "starting", "since": time.time()}
            try:
                if current is not None:
                    self._event(current.name, "swap-out", {"to": name})
                    self._stop(current, drain=True)
                if not os.path.exists(m.binary()):
                    with self.lock:
                        self.state[name] = FAILED
                        self.errors[name] = f"no engine binary at {m.binary()}"
                    self._event(name, "failed", {"error": self.errors[name]})
                    return {"status": FAILED, "error": self.errors[name]}
                with self.lock:
                    self.swap["phase"] = "starting"
                    self.output[name].clear()
                    run = EngineRun(m, self.argv(m), self.fleet.data_dir / "logs" / f"{name}.log", self._line)
                    self.runs[name] = run
                    self.state[name] = STARTING
                    self.errors.pop(name, None)
                    if current is not None:
                        self.swaps += 1
                self._event(name, "start", {"from": current.name if current else None, "argv": run.argv})
                return {"status": STARTING}
            finally:
                with self.lock:
                    self.swap = None

    def unload(self, name: str, drain: bool = True) -> dict:
        m = self.fleet.get(name)
        if m is None:
            raise KeyError(name)
        with self.lock:
            run = self.runs.get(name)
            if self.state[name] == EXTERNAL:
                return {"status": "external"}
            if run is None or not run.alive():
                return {"status": "unloaded"}
        self._stop(m, drain=drain)
        return {"status": "unloaded"}

    def _stop(self, m: Model, drain: bool = True, drain_max: float = 180.0, grace: float = 60.0) -> None:
        with self.lock:
            run = self.runs.get(m.name)
            if run is None or not run.alive():
                self.state[m.name] = UNLOADED
                return
            self.state[m.name] = STOPPING
        self._event(m.name, "stop", {"drain": drain})
        if drain:                                   # let running requests finish, up to drain_max
            end = time.time() + drain_max
            while time.time() < end and run.alive():
                slots = get_json(m.base + "/slots", timeout=2.0)
                if not isinstance(slots, list) or not any(s.get("is_processing") for s in slots):
                    break
                time.sleep(1.0)
        run.signal(signal.SIGTERM)
        end = time.time() + grace
        while time.time() < end and run.alive():
            time.sleep(0.25)
        if run.alive():
            self._line(m.name, f"console: no exit after {grace:.0f} s, killing")
            run.signal(signal.SIGKILL)
            run.proc.wait(timeout=10)
        with self.lock:
            self.state[m.name] = UNLOADED
            self.health[m.name] = None
        self._event(m.name, "stopped", {"exit": run.exit_code})

    def shutdown(self) -> None:
        self.stopping.set()
        for m in self.fleet.models:
            run = self.runs.get(m.name)
            if run and run.alive():
                self._stop(m, drain=False, grace=30.0)


def _loopback(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


def _level(line: str) -> str:
    low = line[:80].lower()
    if "error" in low or "traceback" in low or "fatal" in low:
        return "error"
    if "warn" in low:
        return "warn"
    return "info"
