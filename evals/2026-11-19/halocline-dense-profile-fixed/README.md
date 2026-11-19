# Dense window profile at fixed widths: where a verify window loses to the bandwidth floor (2026-11-19 ~14:10-14:22)

Served code (release 626f0add = aa156951 + the experiment flags `--draft-ahead` and `--draft-min` from local branch
`draft-floor`, not on main; with both off it is aa156951), dense 27B UD-Q4_K_XL, `--long-context --vision cpu`,
`--draft-vocab` 32K (not at T1: refused without the drafter), `--profile`, one fresh server per width:
`--spec T --draft-ahead T-1 --draft-min T-1`, so every window of a run has width T and greedy text repeats. Prose,
thinking off, 1,200 tokens. `profile.py` (tables from `GET /api/logs`), `out/*.profile.txt`, `out/results.json`.
Floors: `stage_bytes.py` sums each stage's tensor bytes in the GGUF the pack copies (15.8 GiB a window); `floor.py`
divides by 1,008 GB/s (spec) -> `floor.txt`. `stage_types.py`: quant mix per stage. `--profile` runs uncaptured with PDL
off (T1 22.0 ms here vs 21.1 served), so read steps and shares. First try `failed-T1-draft-vocab/`.

## Window ms by stage and width (short context)

| stage (floor ms) | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| ffn-up (7.10) | 8.73 | 8.60 | 8.73 | 8.88 | 9.22 | 10.65 |
| **ffn-down (3.69)** | 4.29 | 4.31 | 4.37 | **5.57** | **7.36** | 5.65 |
| gdn-pre (2.56) | 3.39 | 3.40 | 3.43 | 3.51 | 3.63 | 4.30 |
| gdn-out (1.09) | 1.50 | 1.52 | 1.54 | 1.57 | 1.60 | 1.92 |
| attn-pre (0.83) | 1.10 | 1.11 | 1.12 | 1.15 | 1.19 | 1.54 |
| attn-out (0.41) | 0.57 | 0.57 | 0.59 | 0.60 | 0.62 | 0.72 |
| head (1.03) | 1.11 | 1.11 | 1.12 | 1.12 | 1.13 | 1.50 |
| gdn-rec | 0.78 | 0.78 | 0.85 | 0.94 | 1.09 | 1.11 |
| mtp-draft (T-1 steps) | - | 0.65 | 1.29 | 1.94 | 2.59 | 3.18 |
| window (incl. drafting) | 22.02 | 23.19 | 24.23 | 26.59 | 29.84 | 31.95 |

137K: T1 27.40, T4 33.71 ms; the same GEMV numbers (ffn-down 4.29 -> 5.46 at T4) plus attn-core 5.74 -> 7.58.

## What it says

1. **ffn-down's multi-row GEMV breaks at 4 and 5 rows.** Flat to T3 (4.29-4.37), then +1.2 ms at T4 and +3.1 ms at T5;
   at T6 the window switches to the MMQ path (`kMmvqMaxRows = 5`) and it is 5.65. No other stage steps before T6.
   ffn-down is the one matrix with a long reduction (17,408 in, 5,120 out). T4 is the most common prose window
   (`../halocline-w35-confidence-ab`), so this is ~1.2 ms of a ~25 ms round: about the whole short-prose round gap to
   NInfer (25.1 vs 23.8 ms). At T5, MMQ alone would save ~1.7 ms. **First kernel lever.**
2. **Small projections run at 71-76% of bandwidth** (gdn-pre, gdn-out, attn-pre, attn-out: 5,120-row outputs or short
   reductions); ffn-up 81%, ffn-down 86% (T1), the 248K-row Q6_K head 93%. The format mix is the same across stages
   (Q5_K, IQ4_XS, Q6_K, Q4_K), so it is shape, not format. Bringing these to 86% would save ~1.3 ms a window at every
   width. **Second lever.**
3. Drafting costs 0.65 ms a step with the 32K head; gdn-rec 0.8-1.1 ms; launch gap 0.04-0.4 ms. Not where the time is.
