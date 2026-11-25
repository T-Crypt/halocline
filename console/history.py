"""The console's history: every request each engine served, a sample every 10 s, and the fleet's events.

One SQLite file (stdlib only), so the page has numbers for a model while it is not loaded and after a restart.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  model TEXT NOT NULL, rid TEXT NOT NULL, t REAL NOT NULL, path TEXT, finish TEXT,
  prompt_tokens INTEGER, reused INTEGER, output_tokens INTEGER, prompt_ms REAL, decode_tok_s REAL,
  prefill_tok_s REAL, ttft_s REAL, duration_s REAL, queue_s REAL, load_s REAL, hit_rate REAL,
  drafts_offered INTEGER, drafts_accepted INTEGER, slot INTEGER, raw TEXT,
  PRIMARY KEY (model, rid, t)
);
CREATE INDEX IF NOT EXISTS requests_t ON requests (t);
CREATE TABLE IF NOT EXISTS samples (
  t REAL NOT NULL, model TEXT, state TEXT, tok_s REAL, prefill_tok_s REAL, active INTEGER, queued INTEGER,
  ctx_used INTEGER, gpu_util REAL, gpu_power REAL, gpu_temp REAL, vram_used REAL, vram_total REAL,
  clock_sm REAL, pcie_rx REAL, pcie_tx REAL, cpu REAL, ram_used REAL, ram_total REAL, disk_read REAL
);
CREATE INDEX IF NOT EXISTS samples_t ON samples (t);
CREATE TABLE IF NOT EXISTS events (t REAL NOT NULL, model TEXT, kind TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS events_t ON events (t);
"""

REQUEST_COLS = ("path", "finish", "prompt_tokens", "reused", "output_tokens", "prompt_ms", "decode_tok_s", "ttft_s",
                "duration_s", "queue_s", "load_s", "hit_rate", "drafts_offered", "drafts_accepted", "slot")
SAMPLE_COLS = ("t", "model", "state", "tok_s", "prefill_tok_s", "active", "queued", "ctx_used", "gpu_util",
               "gpu_power", "gpu_temp", "vram_used", "vram_total", "clock_sm", "pcie_rx", "pcie_tx", "cpu",
               "ram_used", "ram_total", "disk_read")


class History:
    def __init__(self, path: Path, retention_days: int = 30):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.retention_s = retention_days * 86400
        self.pruned = 0.0

    # ------------------------------------------------------------------ writes
    def requests(self, model: str, rows: list[dict]) -> int:
        """Store the engine's request rows; ones already stored are skipped. Returns how many were new."""
        new = 0
        with self.lock:
            for r in rows:
                if not r.get("id") or not r.get("time"):
                    continue
                t = r.get("timings") or {}
                vals = [r.get(c) for c in REQUEST_COLS]
                pre = t.get("prompt_per_second")
                cur = self.db.execute(
                    f"INSERT OR IGNORE INTO requests (model, rid, t, {', '.join(REQUEST_COLS)}, prefill_tok_s, raw) "
                    f"VALUES (?, ?, ?, {', '.join('?' * len(REQUEST_COLS))}, ?, ?)",
                    [model, r["id"], r["time"], *vals, pre, json.dumps(r)])
                new += cur.rowcount
        return new

    def sample(self, s: dict) -> None:
        with self.lock:
            self.db.execute(f"INSERT INTO samples ({', '.join(SAMPLE_COLS)}) VALUES ({', '.join('?' * len(SAMPLE_COLS))})",
                            [s.get(c) for c in SAMPLE_COLS])
            if s["t"] - self.pruned > 3600:
                cut = s["t"] - self.retention_s
                for table in ("samples", "events", "requests"):
                    self.db.execute(f"DELETE FROM {table} WHERE t < ?", (cut,))
                self.pruned = s["t"]

    def event(self, model: str, kind: str, detail: dict) -> None:
        with self.lock:
            self.db.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (time.time(), model, kind, json.dumps(detail)))

    # ------------------------------------------------------------------ reads
    def _rows(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            cur = self.db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def series(self, since: float, buckets: int = 240, model: str | None = None) -> dict:
        """Samples since `since`, averaged into at most `buckets` points (max for the speed columns' peaks)."""
        now = time.time()
        width = max(10.0, (now - since) / buckets)
        where, args = "t >= ?", [since]
        if model:
            where += " AND model = ?"
            args.append(model)
        rows = self._rows(
            f"SELECT CAST((t - ?) / ? AS INTEGER) AS b, MIN(t) AS t, "
            f"AVG(NULLIF(tok_s, 0)) AS tok_s, MAX(tok_s) AS tok_s_max, AVG(NULLIF(prefill_tok_s, 0)) AS prefill_tok_s, "
            f"AVG(gpu_util) AS gpu_util, AVG(gpu_power) AS gpu_power, MAX(gpu_temp) AS gpu_temp, "
            f"AVG(vram_used) AS vram_used, MAX(vram_total) AS vram_total, AVG(cpu) AS cpu, AVG(ram_used) AS ram_used, "
            f"MAX(ram_total) AS ram_total, MAX(active) AS active, MAX(queued) AS queued, MAX(ctx_used) AS ctx_used, "
            f"MAX(model) AS model FROM samples WHERE {where} GROUP BY b ORDER BY b",
            [since, width, *args])
        return {"since": since, "width": width, "rows": rows}

    def request_rows(self, since: float = 0, model: str | None = None, limit: int = 500) -> list[dict]:
        where, args = "t >= ?", [since]
        if model:
            where += " AND model = ?"
            args.append(model)
        cols = ", ".join(("model", "rid", "t", *REQUEST_COLS, "prefill_tok_s"))
        return self._rows(f"SELECT {cols} FROM requests WHERE {where} ORDER BY t DESC LIMIT ?", [*args, limit])

    def events(self, since: float = 0, limit: int = 200) -> list[dict]:
        rows = self._rows("SELECT t, model, kind, detail FROM events WHERE t >= ? ORDER BY t DESC LIMIT ?",
                          (since, limit))
        for r in rows:
            try:
                r["detail"] = json.loads(r["detail"])
            except (TypeError, ValueError):
                pass
        return rows

    def model_stats(self, since: float = 0) -> dict:
        """Per model: requests, tokens, mean and best speeds, draft acceptance, load times, last use."""
        rows = self._rows(
            "SELECT model, COUNT(*) AS requests, SUM(output_tokens) AS output_tokens, SUM(prompt_tokens) AS prompt_tokens,"
            " SUM(reused) AS reused, AVG(decode_tok_s) AS decode_mean, MAX(decode_tok_s) AS decode_best,"
            " AVG(prefill_tok_s) AS prefill_mean, MAX(prefill_tok_s) AS prefill_best, AVG(ttft_s) AS ttft_mean,"
            " SUM(drafts_accepted) AS drafts_accepted, SUM(drafts_offered) AS drafts_offered, MAX(t) AS last"
            " FROM requests WHERE t >= ? GROUP BY model", (since,))
        out = {r["model"]: r for r in rows}
        for r in self._rows("SELECT model, detail, t FROM events WHERE kind = 'ready' AND t >= ? ORDER BY t", (since,)):
            try:
                load = json.loads(r["detail"]).get("load_s")
            except (TypeError, ValueError):
                load = None
            m = out.setdefault(r["model"], {"model": r["model"]})
            m.setdefault("loads", []).append(load)
            m["loaded_last"] = r["t"]
        for m in out.values():
            loads = [x for x in m.pop("loads", []) if x is not None]
            m["load_count"] = len(loads)
            m["load_mean_s"] = round(sum(loads) / len(loads), 1) if loads else None
            # decode speed spread for the card's sparkline: the last 40 requests, oldest first
            m["recent"] = [r["decode_tok_s"] for r in self._rows(
                "SELECT decode_tok_s FROM requests WHERE model = ? AND decode_tok_s IS NOT NULL ORDER BY t DESC LIMIT 40",
                (m["model"],))][::-1]
        return out
