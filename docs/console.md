# The console

One page and one API port for every model a machine serves with Halocline. Each model runs its own release, so builds
from different branches share the console, one on the card at a time. The page is monitoring only: no chat.

```sh
python3 -m console --fleet examples/console/fleet.toml        # http://127.0.0.1:9785/
python3 -m console --fleet FILE --check                       # print each model's command and exit
```

Standard library only. It holds no VRAM: card readings come from NVML through ctypes, which opens no CUDA context.

## The fleet file

TOML. `[console]` holds where it listens and keeps its history; each `[[model]]` is one model.

| Key | What |
|---|---|
| `name` | the id clients ask for; the engine starts with `--alias NAME` |
| `aliases` | other ids clients may use; the console renames them to `name` before the engine sees them |
| `release` | a checkout or release directory holding `bin/halocline` (`exe` names another binary) |
| `port` | the engine's own loopback port |
| `args` | the engine's arguments after the binary (`serve`, the pack, flags) |
| `label`, `branch`, `notes` | shown on the model's card |
| `lease`, `min_turn`, `priority` | the Berth lease, when the console runs engines under `berth run` |
| `env` | extra environment for the engine |
| `meta` | free-form; the console does not read it (for tools that render client configs) |

[`examples/console/fleet.toml`](../examples/console/fleet.toml) is a two-model example.

## Two modes

**Console mode** (default). The console starts and stops the engines itself. A load ends the running engine after its
requests finish, then starts the next. With `berth = "berth"` in `[console]` each start is wrapped in
`berth run --as LEASE`, and Berth's drain posts back to the console.

**Berth mode** (`gateway = "http://127.0.0.1:9799"`, `gateway_key_file = ...`). Berth starts and stops every engine, each a
Berth `[[engine]]` with `start` and `serves`. Load is `POST /v1/berth/switch`, Unload is `/v1/berth/stop`, and the
console only watches each model's port. Engines outlive a console restart. A request to a model that is down asks Berth
to switch, waits for ready, then goes to the engine, which speaks every API Halocline has.

## Endpoints

| Path | What |
|---|---|
| `GET /` | the page: Overview, Fleet, Engine, History, Hardware, Logs |
| `GET /api/state`, `/api/stream` | the fleet, the model on the card, its metrics and the host; the stream sends only what changed |
| `GET /api/history?range=6h&model=` | samples in buckets, per-model stats, load and swap events (15m to 30d) |
| `GET /api/requests`, `/api/logs`, `/api/models/NAME` | stored request rows, engine logs, one model |
| `POST /api/models/NAME/load`, `/unload` | from the page or a non-browser client; a form on another site is refused |
| `GET /v1/models`, `POST /v1/...` | every model and alias; a request loads its model first |
| `GET /health` | the console's own health; never loads a model |

## History

SQLite in `data` (default `~/.local/state/halocline-console`): every request the engines report, a sample every 10 s
and the load, swap and stop events, kept 30 days (`retention_days`). The cards keep their numbers while a model is
unloaded.

## Cost

One-second readings only while a page is open, an engine generates or a model loads. Otherwise every 5 s, card holders
every 60 s and idle models every 15 s. About 0.2% of one core and 60 MB at idle on the reference machine. An open page
receives about 2 KB a second.

## Install as an app

The page has a manifest, maskable icons and a service worker that caches only the page itself (never `/api` or `/v1`).
Served over HTTPS (a reverse proxy or a tailnet's HTTPS), it installs from the browser menu on Android, iOS and desktop.

Loopback by default. `--api-key-env VAR` makes the page and the API ask for a key.
