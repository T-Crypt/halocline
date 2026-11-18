# Halocline sparse non-regression gate: release b22ecbb2 against d314a21a, 2026-11-17 (the workstation, RTX 4090, the board)

The dense D8 work (the rename, the KV format enum, the planner, the drafter levers, the q4_0 code moved to shared files)
must not cost the sparse path speed or quality. REF is the served sparse release `d314a21a` (`bin/basalt`); NEW is the
release tree of `d8-integration` at `b22ecbb2` (`bin/halocline`). Both read the same pack
(`QFN-UD-IQ4_XS-v3.1-compat-mtp`) and are scored against `flashnext-bf16-base.kld`. One engine on the card at a time
under a Berth training lease, 350 W enforced, driver 615.78.08. Runner: `run.sh`; its own verdicts: `summary.md`.

| # | Gate | Result |
|---|---|---|
| 1 | Greedy identity, `--spec 0`, 900 tokens, pinned placement | **PASS**: GREEDY IDENTICAL (`1-greedy/greedy.txt`) |
| 2 | KLD (100 chunks) and speed, `--kv int8 --spec 8` | **PASS** after the 131K rerun below: quality byte-identical, speed within 1% |
| 3 | Sampling distribution, temperature 1.0, 24 prompts x 32 seeds | **PASS**: top-1, top-10, log-prob and first-token TV all inside REF's own seed-to-seed spread (`3-sampling/compare.txt`) |
| 4 | Tool-call breadth, 24 labelled cases, temperature 0 | **PASS**: REF 24/24, NEW 24/24, agreement 24/24 (`4-tools/`) |

## Gate 2 in detail

Quality, the two complete runs (`2-kv/paired-ref1-new2.md`): PPL 3.2323, mean KLD 0.08195, top-1 90.45% on both, the
published numbers; the two `rows.tsv` are byte-identical, so the paired difference is exactly zero.

Speed, from the same runs:

| | REF | NEW | NEW - REF |
|---|---|---|---|
| conv / struct / prose decode | 135.2 / 266.2 / 137.0 | 134.9 / 264.5 / 136.8 | -0.2 / -0.6 / -0.1% |
| prefill 32K, all four runs | 5,785, 5,848 | 5,846, 5,826 | within 1% |

**The first 131K prefill arms were invalid on both builds.** `new1` and `ref2` were stopped by the engine's own stall
guard at about token 133,198 ("no progress for 60 s"), and the two that finished read 1,841 and 2,428 t/s against the
published 6,019. The engine log at the stall: 53,301 MiB resident, 4,053 MiB of RAM available, the 4 GiB compressed swap
full. The host was short of memory (other sessions and a long-running background job), not either build.

**The rerun (`2b-pp131k/`)**, prefill only, ABBA, after stopping the background jobs and pausing the periodic timers
(`memory-before.txt`: 59,353 MiB available):

| Run | REF | NEW |
|---|---|---|
| a | 5,878 | 5,892 |
| b | 5,994 | 5,948 |
| mean | 5,936 | 5,920 (**-0.28%**) |

Within the gate's 3% band, and within 1.6% of the published 6,019. The spread between the two halves of each build
(2.0%, 0.9%) is larger than the difference between builds.

## What this does not cover

- Depth: the sparse gate scores 512-token chunks and prefill at 32K / 131K, the published protocol.
- The sparse served entry still runs `d314a21a` until its llama-swap smoke on `b22ecbb2`.
