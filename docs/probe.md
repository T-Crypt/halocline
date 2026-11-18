# halocline-probe

`bin/halocline-probe` starts `halocline --serve` as a child process, asks it for the machine's rates, measures cold
decode speed on the three standard fixtures, tries a few settings, and prints a short report. It is manual: it writes
no profile and saves no setting, and it stops the engine it started.

```
halocline-probe --model PACK.halo [-- ENGINE ARGS...]
```

| Option | Meaning |
|---|---|
| `--model FILE` | the pack to measure; required |
| `--engine BIN` | the engine binary; by default `halocline` beside the probe, else the one the build names |
| `--fixtures DIR` | the fixture ids; by default the release's `data/bench`, or the model's own folder inside it |
| `--logs DIR` | keep the engine log there; without it a temporary directory holds it and is removed at exit |
| `--gpu A,B,...` or `auto` | the cards, chosen before the engine starts, as for the server; default GPU 0 alone |
| `-- ARGS...` | engine flags, as for `halocline-bench` |

The probe sizes `--max-context` to fit the fixtures unless the engine arguments set one. An
`--expert-profile-save FILE` after `--` is compared, never written.

## What it does

1. Starts the engine and sends `PROBE rates`: GPU facts and memory bandwidth, the PCIe link, host memory bandwidth by
   worker count, the CPU expert time per worker count, and the cost of one expert-sized host-to-device copy.
2. Runs `conv`, `struct` and `prose` once each, 400 tokens, temperature 0. These are cold speeds: the first request
   after load counts. The token budget is part of "cold", because the tiers keep adapting, so a longer request reads
   faster.
3. Measures the suggestions below and prints the report.

A suggestion is printed only when the measurement clears the tool's own margin: 2% for the pool, 8% for the prefill
share, and the measured noise band for the adaptive settings and the saved profile.

- **Pool workers.** The fewest workers within 2% of the fastest measured CPU expert time.
- **Prefill CPU share.** Only when the model takes `--prefill-cpu-share`: 8192-token chunks at share 0, 1, the
  balancing share and the default, printed when the balance is at least 8% faster.
- **Adaptive tier.** Only when the model takes `--adapt-swaps`: `every 64 / swaps 256 / decay 0.7`, `1 / 160` and
  `0 / 12` are run from the start placement on a prose-then-struct sequence, against the engine's own setting, which
  runs twice to set the noise band.
- **Saved expert profile.** The profile given after `--` is run against the engine's own start; a line names the winner
  when the difference clears the band.

The report has one line for the GPU memory bandwidth, PCIe and RAM, one for the expert tiers, KV format and context,
the cold decode rates for `prose`, `conv` and `struct`, the adaptive test's runs when it ran, each suggestion, the
engine flags the run used, the wall time, and the line "Measured on this machine. Hand-tuned settings can still beat
these."

## What a pack must provide

`PROBE rates` needs a session that implements `probe_env()` and a RAM-resident expert source. Today that is the
`qwen4exp` (Flash-Next) model; a dense `qwen35` session answers "this model has no probe" and the probe exits 1.

The live commands also differ by path. The Flash-Next runtime accepts `PROBE set workers=K share=F` and `PROBE adapt`.
The shared dense serve path handles `PROBE set workers=K`; its `PROBE rates` answers "this model has no probe".

## Engine commands

`PROBE rates` answers between requests with `PROBE begin rates v1`, one `PROBE <section> key=value` line per fact, and
`PROBE end ms=`. Its sections are `gpu`, `pack`, `link`, `gpu_mem`, `host`, `host_mem`, `cpu_expert`, `copy`, and
`step` (the time each step took). A measurement the pack cannot support answers `n/a` with a reason. The command
restores the CPU pool's worker count and frees every buffer it allocates before it returns.

`PROBE set workers=K` sets the CPU pool's worker count for the next requests, and `share=F` sets the prompt CPU share
where the model has one. `PROBE adapt [every=N] [swaps=N] [decay=F] [profile=FILE]` resets the adaptive tiers and
sets their settings for the next request, where the model has them.
