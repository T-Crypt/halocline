"""The collector: reads the model on the card once a second and the host always.

The engine's /metrics carries its shape, live state, request rows, bests and totals; /slots each slot's progress. The
host readings come from serve.util.telemetry (NVML through ctypes: no CUDA context, no VRAM), so the page shows the
card while no engine is loaded. Every few seconds it lists the processes holding the card, so a squatter shows up.
"""
from __future__ import annotations

import collections
import shutil
import subprocess
import threading
import time

from console.supervisor import Supervisor, get_json
from serve.util.telemetry import Telemetry

LIVE_POINTS = 300            # 5 minutes of one-second points for the live charts
SAMPLE_EVERY = 10.0          # seconds between history rows
APPS_EVERY = 5.0             # card holders while a page watches
APPS_IDLE = 60.0             # and while none does
IDLE_TICK = 5.0              # the engine and host while no page watches and nothing generates


class Collector:
    def __init__(self, sup: Supervisor, history=None, telemetry: Telemetry | None = None):
        self.sup, self.history = sup, history
        if telemetry is None:
            telemetry = Telemetry()
            telemetry.stop()             # its own 1 Hz thread: the collector samples on its own clock instead
        self.tel = telemetry
        self.host: dict = {}
        self.viewers = 0                 # open /api/stream connections: a page watching gets one-second readings
        self.lock = threading.Lock()
        self.metrics: dict[str, dict] = {}        # the last /metrics each model answered, kept after it unloads
        self.metrics_at: dict[str, float] = {}
        self.slots: dict[str, list] = {}
        self.engine_logs: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=3000))
        self.log_after: dict[str, int] = {}
        self.live = collections.deque(maxlen=LIVE_POINTS)   # one row per second: speeds and the host
        self.apps: list[dict] = []
        self.apps_at = 0.0
        self.sampled = 0.0
        self.seen: dict[str, set] = collections.defaultdict(set)    # request ids already turned into prefill points
        self.prefill_done: list[tuple[float, float]] = []            # (time, t/s) of requests finished since the sample
        self.stopping = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="collector").start()

    # ------------------------------------------------------------------ the loop
    def _loop(self):
        while not self.stopping.is_set():
            t0 = time.time()
            try:
                self.tick(t0)
            except Exception as e:  # noqa: BLE001 - collection must never stop
                self.sup._line(self.sup.active() or self.sup.fleet.models[0].name, f"console: collector: {e}")
            busy = (self.live[-1].get("tok_s") or 0) > 0 if self.live else False
            fast = self.viewers > 0 or busy or self.sup.transitioning()
            self.stopping.wait(max(0.05, (1.0 if fast else IDLE_TICK) - (time.time() - t0)))

    def tick(self, now: float) -> None:
        name = self.sup.active()
        m = self.sup.fleet.get(name) if name else None
        metrics = slots = None
        if m is not None and self.sup.state[name] in ("ready", "loading", "external", "stopping"):
            metrics = get_json(m.base + "/metrics", timeout=2.0)
            slots = get_json(m.base + "/slots", timeout=2.0)
            if self.viewers:
                self._pull_logs(m)
        with self.lock:
            if isinstance(metrics, dict) and "engine" in metrics:
                self.metrics[name] = metrics
                self.metrics_at[name] = now
            if isinstance(slots, list):
                self.slots[name] = slots
        if isinstance(metrics, dict) and self.history is not None and metrics.get("requests"):
            self.history.requests(name, metrics["requests"])
        # A prompt read is usually over between two readings (a cached prompt reads in milliseconds), so live.phase
        # rarely says prefill. Each finished request reports its own read rate: that is the prefill point.
        finished = self._new_prefill(name, metrics) if isinstance(metrics, dict) else []
        if hasattr(self.tel, "sample"):
            try:
                self.host = self.tel.sample()
            except Exception:  # noqa: BLE001 - host readings must never stop collection
                pass
            host = self.host
        else:
            host = self.tel.snapshot()["now"]
        live = (metrics or {}).get("live") or {}
        gpu = (host.get("gpus") or [{}])[0]
        slot_rows = slots if isinstance(slots, list) else []
        row = {
            "t": now, "model": name, "state": self.sup.state.get(name) if name else None,
            "tok_s": live.get("tok_s") or 0.0,
            "prefill_tok_s": (live.get("prefill_tok_s_mean") if live.get("phase") == "prefill" else None)
                             or (max(finished) if finished else 0.0),
            "active": (live.get("slots") or {}).get("active"), "queued": live.get("queued"),
            "ctx_used": max((s.get("ctx_used") or 0 for s in slot_rows), default=None),
            "gpu_util": gpu.get("util"), "gpu_power": gpu.get("power"), "gpu_temp": gpu.get("temp"),
            "vram_used": gpu.get("mem_used"), "vram_total": gpu.get("mem_total"), "clock_sm": gpu.get("clock_sm_mhz"),
            "pcie_rx": gpu.get("pcie_rx_mb"), "pcie_tx": gpu.get("pcie_tx_mb"), "cpu": host.get("cpu"),
            "ram_used": host.get("ram_used"), "ram_total": host.get("ram_total"), "disk_read": host.get("disk_read_mb"),
            "slot_tok_s": [s.get("tok_s") or 0.0 for s in slot_rows],
        }
        with self.lock:
            self.live.append(row)
        if self.history is not None and now - self.sampled >= SAMPLE_EVERY:
            self.sampled = now
            recent = [r for r in self.live if r["t"] > now - SAMPLE_EVERY]
            agg = dict(row)
            for k in ("tok_s", "prefill_tok_s"):
                vals = [r[k] for r in recent if r.get(k)]
                agg[k] = sum(vals) / len(vals) if vals else 0.0
            if self.prefill_done:                       # requests that finished since the last sample, between polls too
                agg["prefill_tok_s"] = sum(v for _, v in self.prefill_done) / len(self.prefill_done)
                self.prefill_done = []
            self.history.sample(agg)
        if now - self.apps_at >= (APPS_EVERY if self.viewers else APPS_IDLE):
            self.apps_at = now
            self.apps = gpu_apps()

    def _new_prefill(self, name: str, metrics: dict) -> list[float]:
        """Read rates (t/s) of the requests this engine finished since the last reading. The first reading after the
        console starts only marks what is already there, so old requests do not show as a burst."""
        seen = self.seen[name]
        first = not seen
        out = []
        for r in metrics.get("requests") or []:
            key = (r.get("id"), r.get("time"))
            if key in seen:
                continue
            seen.add(key)
            rate = (r.get("timings") or {}).get("prompt_per_second")
            if not first and rate and (r.get("timings") or {}).get("prompt_n"):
                out.append(float(rate))
        if len(seen) > 2000:
            seen.clear()
            seen.update((r.get("id"), r.get("time")) for r in metrics.get("requests") or [])
        self.prefill_done += [(time.time(), v) for v in out]
        return out

    def _pull_logs(self, m) -> None:
        after = self.log_after.get(m.name)
        url = m.base + "/api/logs" + (f"?after={after}" if after is not None else "?limit=400")
        got = get_json(url, timeout=2.0)
        if not isinstance(got, dict):
            return
        lines = got.get("lines") or []
        last = got.get("last")
        if after is not None and isinstance(last, int) and last < after:      # the engine restarted: ids began again
            self.log_after.pop(m.name, None)
            return
        with self.lock:
            buf = self.engine_logs[m.name]
            for ln in lines:
                buf.append(ln)
        if isinstance(last, int):
            self.log_after[m.name] = last

    # ------------------------------------------------------------------ reads
    def snapshot(self, points: int = 120) -> dict:
        with self.lock:
            live = list(self.live)[-points:]
            name = self.sup.active()
            metrics = self.metrics.get(name) if name else None
            return {"live": live, "metrics": metrics, "metrics_model": name if metrics else None,
                    "slots": self.slots.get(name, []) if name else [], "apps": self.apps,
                    "static": self.tel.static}

    def model_metrics(self, name: str) -> dict | None:
        with self.lock:
            m = self.metrics.get(name)
            return {"metrics": m, "at": self.metrics_at.get(name)} if m else None

    def logs(self, name: str, after: int = 0, limit: int = 400) -> list[dict]:
        with self.lock:
            rows = [r for r in self.engine_logs.get(name, ()) if (r.get("id") or 0) > after]
        return rows[-limit:]


def gpu_apps() -> list[dict]:
    """The processes holding the card (nvidia-smi's compute apps): pid, name, MiB. [] without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                mib = int(parts[2])
            except ValueError:
                mib = None
            apps.append({"pid": int(parts[0]), "name": parts[1].rsplit("/", 1)[-1], "path": parts[1], "mib": mib,
                         "cmd": _cmdline(int(parts[0]))})
    return apps


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            words = f.read().split(b"\0")
    except OSError:
        return ""
    words = [w.decode("utf-8", "replace") for w in words if w]
    return " ".join(w.rsplit("/", 1)[-1] if i == 0 else w for i, w in enumerate(words[:4]))
