# Adaptive expert tier

Flash-Next only. The card holds part of the experts. The adaptive tier counts what the conversation routes to and, in
rounds beside decode, moves the most used experts into VRAM. The dense path has no experts and no such tier.

## How a round works

Routing counts rank the experts not in VRAM. A round swaps the most-routed ones in for the least-routed resident ones.
A round takes three window boundaries: the evicted experts are copied out to the RAM copy, the incoming ones are
copied in on a side stream, then they are used (`src/models/qwen4exp/serve/tiers.cpp:417`). Counts decay after each
round. Every layer ranks together when their expert blobs are one size. Moving experts changes rounding, so output with
adaptation on is not bit-identical run to run.

## Settings

| Setting | Default | Trade-off |
| --- | --- | --- |
| `adapt-every` | 1 on a server, 4 outside `--serve` | Fewer, later rounds cost less link, but a topic change or a fresh engine waits N windows. |
| `adapt-swaps` | 96 | Bigger rounds catch up faster and move more bytes. The RAM copy keeps four times N free places. |
| `adapt-decay` | 0.85 with the resident RAM copy (a server's default), 0.7 otherwise | Lower follows topic changes sooner. Higher keeps a long-run placement. |
| `expert-profile-save` | off | Saves the placement; a continuing workload starts on it, a different one starts wrong. `--expert-profile-save-every` sets the period (default 10 minutes, 0 = on exit only). |

`hit=` in the `done` line is the share of routed experts the VRAM cache held, not the CPU or SSD; compare `tg` for the
effect of a placement, not `hit`.

There is no measured tuning run for this tier in the repo. The defaults are the ones the engine ships, and
[halocline-probe](../probe.md) tries the swap counts on your machine from a clean start placement.
