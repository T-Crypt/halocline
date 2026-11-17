# Halocline P0 speed floors, 2026-11-16 (the workstation, RTX 4090, ticket the board)

Runner `run.sh`: Strata 0.2.10 + DFlash2 on UD-IQ4_XS v3.1 (`~/strata/strata-ud-iq4_xs.json`, q4_0 KV), 3 arms of
`lsw.py smoke --long` + `strata bench run` (standard-v1); then NInfer-Qwen3.8-27B through llama-swap, 3 x `lsw.py smoke
--long`. One engine on the card at a time. Timers paused from run 3 on.

## Strata 0.2.10, DFlash2 (sparse floor)

| Measure | Run 1 | Run 3 | 2026-11-04 (same config) |
|---|---|---|---|
| warm decode, smoke | 96.8-99.9 | 90.6-101.5 | |
| standard-v1 short decode (median) | 86.7 | 85.9 | 90.9 |
| standard-v1 4K decode / prompt | 77.4 / 798 | 82.5 / 1297 | 80.6 |
| standard-v1 16K decode / prompt | 88.5 / 1303 | 75.7 / 1290 | 83.2 |
| 36K needle decode / prompt | 69.9 / 1504 | 70.5 / 1402 | 71.7 / about 2410 |
| 137K long decode / prompt | 95.6 / 2280 | 92.9 / 2079 | 82.3 / about 2870 |

Run 2: killed by earlyoom at 3.6% free RAM during the 137K prefill (engine RSS 49.2 GiB, 15:23:47); its 36K needle had
passed at 69.4 decode / 1494 prompt. Rerun of its long step: `strata/2r-df2.txt`.

Prompt speed at 36K and 137K reads 30 to 40% below the 2026-11-04 notes. Same release, config and quant; the difference
is not explained yet (RAM pressure from a 5 GiB `/tmp` and timers during runs 1-2 is one candidate, run 3 ran with
timers paused and is no faster). Treat the floor as the lower of the two dates until re-measured on a quiet host.

## NInfer-Qwen3.8-27B (dense floor)

| Measure | Run 1 | Runs 2-3 |
|---|---|---|
| warm decode | 112.5-118.3 | 112.4-118.3 |
| 36K needle decode / prompt | 118.0 / 3215 | 118.3 / cache hit |
| 137K long decode / prompt | 93.0 / 2155 | 93.0 / cache hit |

Runs 2 and 3 hit NInfer's prefix cache (137K "read" in about 1 s), so only run 1's prompt rates count. Decode is
consistent across all three. The 2026-10-03 "about 130 t/s decode" figure was a different bench; this smoke's warm
decode is 112-118.

## Floors for Halocline (spec §9.4 refined)

- Sparse: decode 86 short, 77 at 4K, 70 at 36K, 93 at 137K; prompt 1,290 at 16K, 1,400 at 36K, 2,080 at 137K.
- Dense: decode 112 short, 118 at 36K, 93 at 137K; prompt 3,215 at 36K, 2,155 at 137K.

## Rerun on a quieter host (15:40, after the memory cleanup)

Run 2's long step rerun with timers paused, `/tmp` scratch freed (5 GiB to 1.3 GiB) and swappiness 10:
`strata/2r-df2.txt`: warm decode 94.6-100.5; 36K needle 68.1 decode / **2,403** prompt; 137K long 86.6 decode /
**2,864** prompt. Prompt rates match the 2026-11-04 notes (about 2,410 and 2,870). **The 30-40% prompt loss in runs 1-3
was host memory pressure, not the engine.** The sparse prompt floors are therefore 2,400 at 36K and 2,860 at 137K.
