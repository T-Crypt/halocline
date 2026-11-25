"""The fleet file: every model the console can run, each with the release that serves it.

A fleet is TOML:

    [console]
    host = "127.0.0.1"          # where the console listens
    port = 9797
    data = "~/.local/state/halocline-console"   # history database and engine logs
    berth = "berth"             # optional: run each engine under `berth run` (a GPU lease)

    [[model]]
    name    = "Qwen-Sparse"     # the id clients ask for; the engine is started with --alias NAME
    label   = "Sparse 80B-A3B"  # shown on the page (default: name)
    release = "~/halocline/releases/aa156951"   # a checkout or release dir holding bin/halocline
    port    = 8092              # the engine's own loopback port
    args    = ["serve", "/models/sparse.halo", "--spec", "8"]
    lease   = "halocline"       # optional: the berth engine name this model leases the card as
    branch  = "main"            # optional, shown on the card

The console adds --host 127.0.0.1 --port PORT --alias NAME to args when they are not already there. Each model can name
its own release, so builds from different branches run side by side, one at a time.
"""
from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PORT = 9797


@dataclass
class Model:
    name: str
    release: str
    port: int
    args: list[str]
    label: str = ""
    lease: str | None = None
    branch: str | None = None
    priority: str = "foreground"
    min_turn: str | None = None
    notes: str = ""
    env: dict[str, str] = field(default_factory=dict)
    exe: str | None = None                      # the engine binary; default RELEASE/bin/halocline
    aliases: list[str] = field(default_factory=list)   # other names clients may ask for (renamed to `name` upstream)
    engine: str | None = None                   # gateway mode: the berth engine that serves it (default: lease)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def binary(self) -> str:
        return self.exe or str(Path(self.release) / "bin" / "halocline")

    def engine_argv(self) -> list[str]:
        argv = [self.binary(), *self.args]
        if "--host" not in self.args:
            argv += ["--host", "127.0.0.1"]
        if "--port" not in self.args:
            argv += ["--port", str(self.port)]
        if "--alias" not in self.args:
            argv += ["--alias", self.name]
        return argv

    def public(self) -> dict:
        """What the page shows about the model's setup (no environment, which may hold paths a user keeps private)."""
        return {"name": self.name, "label": self.label or self.name, "release": Path(self.release).name,
                "aliases": list(self.aliases),
                "branch": self.branch, "port": self.port, "lease": self.lease, "notes": self.notes,
                "args": shlex.join(self.args)}


@dataclass
class Fleet:
    models: list[Model]
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    data: str = "~/.local/state/halocline-console"
    berth: str | None = None
    berth_config: str | None = None
    retention_days: int = 30
    title: str = "Halocline"
    gateway: str | None = None                  # berth gateway URL: berth starts and stops the engines, not the console
    gateway_key_file: str | None = None

    def resolve(self, name: str) -> Model | None:
        """The model a client's name means: its own name or one of its aliases."""
        for m in self.models:
            if name == m.name or name in m.aliases:
                return m
        return None

    def gateway_key(self) -> str | None:
        if not self.gateway_key_file:
            return None
        try:
            return Path(os.path.expanduser(self.gateway_key_file)).read_text().strip() or None
        except OSError:
            return None

    def get(self, name: str) -> Model | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    @property
    def data_dir(self) -> Path:
        return Path(os.path.expanduser(self.data))


def _expand(p: str, here: Path) -> str:
    p = os.path.expanduser(p)
    return str(p if os.path.isabs(p) else (here / p).resolve())


MODEL_KEYS = {"name", "label", "release", "port", "args", "lease", "branch", "priority", "min_turn", "notes", "env",
              "exe", "aliases", "engine"}
CONSOLE_KEYS = {"host", "port", "data", "berth", "berth_config", "retention_days", "title", "gateway", "gateway_key_file"}


def parse(raw: dict, here: Path) -> Fleet:
    con = raw.get("console", {})
    extra = set(raw) - {"console", "model"}
    if extra:
        raise ValueError(f"unexpected section {sorted(extra)[0]!r}")
    if not isinstance(con, dict):
        raise ValueError("[console] must be a table")
    bad = set(con) - CONSOLE_KEYS
    if bad:
        raise ValueError(f"[console] {sorted(bad)[0]!r}: not a console setting")
    entries = raw.get("model", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("the fleet needs at least one [[model]]")
    models, names, ports = [], set(), {}
    for i, e in enumerate(entries):
        where = f"model[{i}]"
        bad = set(e) - MODEL_KEYS
        if bad:
            raise ValueError(f"{where}.{sorted(bad)[0]}: not a model setting")
        for k in ("name", "release", "port", "args"):
            if k not in e:
                raise ValueError(f"{where}: {k!r} is required")
        if not isinstance(e["args"], list) or not all(isinstance(a, str) for a in e["args"]):
            raise ValueError(f"{where}.args: a list of strings")
        if not isinstance(e["port"], int) or not 0 < e["port"] < 65536:
            raise ValueError(f"{where}.port: a TCP port")
        aliases = e.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"{where}.aliases: a list of strings")
        for n in (e["name"], *aliases):
            if n in names:
                raise ValueError(f"{where}: {n!r} is named twice")
            names.add(n)
        if e["port"] in ports:
            raise ValueError(f"{where}: port {e['port']} is also {ports[e['port']]!r}'s")
        ports[e["port"]] = e["name"]
        m = Model(name=e["name"], release=_expand(e["release"], here), port=e["port"],
                  args=[os.path.expanduser(a) if a.startswith("~/") else a for a in e["args"]],
                  label=e.get("label", ""), lease=e.get("lease"), branch=e.get("branch"),
                  priority=e.get("priority", "foreground"), min_turn=e.get("min_turn"), notes=e.get("notes", ""),
                  env={str(k): str(v) for k, v in e.get("env", {}).items()},
                  exe=_expand(e["exe"], here) if e.get("exe") else None, aliases=list(aliases),
                  engine=e.get("engine"))
        models.append(m)
    f = Fleet(models=models, **{k: con[k] for k in con if k not in ("berth_config", "gateway_key_file")})
    if con.get("gateway_key_file"):
        f.gateway_key_file = os.path.expanduser(con["gateway_key_file"])
    if f.gateway and f.berth:
        raise ValueError("[console] gateway and berth: choose one (the gateway already leases what it starts)")
    if con.get("berth_config"):
        f.berth_config = _expand(con["berth_config"], here)
    if f.port in ports:
        raise ValueError(f"[console] port {f.port} is also {ports[f.port]!r}'s engine port")
    return f


def load(path: str) -> Fleet:
    p = Path(path)
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    return parse(raw, p.resolve().parent)
