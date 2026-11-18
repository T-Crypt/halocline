# Diagnostics and measurement flags

Flags for measuring, debugging and testing the engine. All are off by default and are not settings.

Binary: **CLI** = `bin/halocline generate` (one-shot, `--tokens IDS`), **serve** = the engine under the server or
`--serve`, **bench** = `halocline-bench`. "Flash-Next" marks a flag the sparse adapter takes; the dense path rejects it
as an unknown argument.

## One-shot generation

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--tokens`, `--tokens-file` | CLI | required | The prompt as token ids. |
| `--max-new` | CLI | 16 | Tokens to generate. |
| `--greedy` | CLI | on | Argmax decoding. |
| `--seed` | CLI | | Sample with this seed under the default `--top-k` 20, `--top-p` 0.95, `--temperature` 1.0. |
| `--stop-eos` | CLI | off | Stop at an end-of-turn token. |
| `--eos-ids` | engine | the pack's | The end-of-turn tokens; implies `--stop-eos`. |
| `--gpu-plan` | engine | | Print the `--gpu` choice as JSON and exit. |

## Timing and profiles

| Flag | Binary | What it does |
|---|---|---|
| `--stats` | CLI | Flash-Next. Per-token wall time, host phases and the pool's breakdown. |
| `--no-pool` | CLI | Flash-Next. Skip the CPU expert pool: the GPU-only floor. |
| `--no-capture` | CLI | Flash-Next. Run layers directly, not as graphs. Needs `--no-pool`. |
| `--stage-timing` | CLI | Flash-Next. Per-stage shares by kernel count. Needs `--no-capture`. |
| `--graph-only` | CLI | Flash-Next. Time the per-layer `pre` graphs only, not the GPU floor. |
| `--gpu-only-full` | CLI | Flash-Next. Replay every graph and the head, no pool: the per-token GPU floor. |
| `--gpu-stages` | CLI | Flash-Next. Time each layer's three graphs from outside the capture, print, exit. |
| `--profile` | serve | GPU stage times of the decode windows. Both paths. It runs the window uncaptured with PDL off, so decode is 3 to 8% below the served build: compare shares and step changes, not absolute ms ([`halocline-dense-profile`](../../evals/2026-11-18/halocline-dense-profile/)). |
| `--wait-stats` | serve | Flash-Next. Host wait times in the decode loop. |
| `--decode-timing` | serve | Flash-Next. One line per request on where decode time went. |
| `--draft-log FILE` | serve | Both paths. One line per verify window (width, accepted, time). |

## Dumps and speculation tests

| Flag | Binary | What it does |
|---|---|---|
| `--dump-residual`, `--dump-mixed`, `--dump-final-r` | CLI | Flash-Next. Write residual or head inputs as f32, for bisecting. |
| `--dump-routing` | CLI | Flash-Next. Routed expert ids and weights per layer and position. |
| `--spec-oracle FILE` | CLI | Flash-Next. With `--spec N` on a pack without a drafter: draft from an expected continuation. The verify exactness test. |
| `--prefill-until P` | CLI | Flash-Next. Batch the first P prompt positions; run the rest teacher-forced. |
| `--spec-corrupt N` | CLI | Flash-Next. Replace every Nth draft with a wrong token, to test rejection. |
| `--force-zero-doorbell` | serve | Flash-Next. Scheduling test on a card that cannot hold every expert. The output is wrong by design. |

## Experts and n-gram table

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--ple-io direct`, `ram` | engine | `direct` | Flash-Next. Read the n-gram table unbuffered (`direct`) or lock it in RAM (`ram`). |
| `--ple-row-cache`, `--ple-inflight`, `--ple-delay-us` | engine | 1048576 rows, 256, 0 | Flash-Next. Row cache size, outstanding reads, and injected read delay for tests. |

## Quality gate and bench

| Flag or key | Binary | Default | What it does |
|---|---|---|---|
| `--quality [N]` | bench | 20 chunks | Run the PPL/KLD/top-1 gate through decode, adaptation off and `--spec 5`. Alone it runs only the gate; with `-p` or `-t`, after them. [method](../precision/quality.md) |
| `--kld-base FILE` | bench | | The llama.cpp KL-divergence base. |
| `--quality-t N` | bench | 3 | Tokens per verify window in the gate. |
| `--ppl-base`, `--ppl-out` | engine | | Base file for `ppl=` requests, and where they write rows. |
| `ppl=K`, `ppl_t=T` | request keys | | Score chunk K; window width T. |

| Bench flag | Default | What it does |
|---|---|---|
| `-t conv,struct,prose` | | Decode fixtures, run in that order. |
| `-p DEPTHS` | `-p 16384` when neither `-t` nor `-p` is given | Prefill depths, e.g. `32k`. |
| `-r N` | 1 | Repetitions; above 1 the table shows mean and deviation. |
| `--seed N` | | Seed every decode request. |
| `--set "FLAG VALUE"` | | Replace or add engine arguments (repeatable). |
| `--engine PATH`, `--fixtures DIR` | `halocline` beside the bench, `data/bench` | Binary and fixtures. |
| `-o md` or `json` | md | Output format. |
| `--power-label W` | read from the GPU | The power limit to print. |
| `--logs DIR`, `--no-windows`, `--print-command` | | Logs, window stats, command printing. |
| `--hardware` | | Measure CPU, RAM and each GPU's link, then exit. |
| `--community`, `--submitter NAME`, `--community-file PATH` | bench | The community preset and its row, which names the model ([community runs](../community.md)). |

`halocline-device` (in the build folder, not shipped): `--list-devices`, `--link-rates`, `--selftest`.

## Experiment keys (`--exp`)

`--exp NAME=VALUE[,...]` sets experiment switches. They carry no stability promise and go away with their experiment.

- `concur9_logit_hash=1`: print a hash of each accepted row's head logits on stderr.
- `concur5_digest=1`: batch slots print a state digest per window.
- `spec_log=1`: with `--draft-log`, also write each row's target distribution.
- `host_ev=1`: print GPU time per window's verify, commit and draft, and the idle gap.
