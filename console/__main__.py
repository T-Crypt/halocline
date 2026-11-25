"""python -m console --fleet FLEET.toml: the Halocline console, one page for every model in the fleet."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

DRAIN_S = 600          # how long a stop waits for passed-through requests to finish


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="halocline-console", description=__doc__)
    ap.add_argument("--fleet", required=True, metavar="FILE", help="the fleet file (TOML): see console/fleet.py")
    ap.add_argument("--host", help="listen address (default: the fleet's, else 127.0.0.1)")
    ap.add_argument("--port", type=int, help="listen port (default: the fleet's, else 9797)")
    ap.add_argument("--api-key-env", metavar="VAR", help="environment variable holding a key the page and API require")
    ap.add_argument("--check", action="store_true", help="read the fleet, print each model's command and exit")
    a = ap.parse_args(argv)

    from console import fleet as fleet_mod
    try:
        fleet = fleet_mod.load(a.fleet)
    except (OSError, ValueError) as e:
        print(f"halocline-console: {a.fleet}: {e}", file=sys.stderr)
        return 2
    if a.host:
        fleet.host = a.host
    if a.port:
        fleet.port = a.port

    from console.supervisor import Supervisor
    if a.check:
        probe = Supervisor.__new__(Supervisor)
        probe.fleet = fleet
        for m in fleet.models:
            ok = "ok" if os.path.exists(m.binary()) else "MISSING BINARY"
            print(f"{m.name} [{ok}] :{m.port}\n  {' '.join(Supervisor.argv(probe, m))}")
        return 0

    key = os.environ.get(a.api_key_env) if a.api_key_env else None
    if a.api_key_env and not key:
        print(f"halocline-console: ${a.api_key_env} is empty", file=sys.stderr)
        return 2
    if fleet.host not in ("127.0.0.1", "localhost", "::1") and not key:
        print("halocline-console: note: listening beyond loopback with no API key", file=sys.stderr)

    from console.server import Console, serve
    con = Console(fleet, api_key=key)
    httpd = serve(con, fleet.host, fleet.port)
    print(f"halocline-console: {len(fleet.models)} models, http://{fleet.host}:{fleet.port}/", flush=True)

    def stop(*_):
        # stop taking requests, then let the ones being answered finish (a deploy must not cut a long prompt off)
        def drain():
            httpd.shutdown()
            end = time.time() + DRAIN_S
            while con.inflight and time.time() < end:
                time.sleep(0.5)
            done.set()
        threading.Thread(target=drain, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    done = threading.Event()
    try:
        httpd.serve_forever(poll_interval=0.5)
        done.wait(DRAIN_S + 5)
    finally:
        if con.inflight:
            print(f"halocline-console: stopping with {con.inflight} request(s) still answering", file=sys.stderr)
        con.sup.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
