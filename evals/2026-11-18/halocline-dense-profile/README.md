# Dense decode window profile, UD-Q4_K_XL K8V4 served flags (Halocline roadmap D5)

2026-11-18 08:35 to 08:41 MST, aphotic, RTX 4090. Halocline branch `dense-profile` c10dd83a (W13; diagnostics only,
the off path is the old code), `bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --spec 8 --max-context 175616
--kv k8v4 --mtp-kv int8 --vision cpu --profile`, under `berth train --as halocline-dense-profile`. `profile.py` starts a
fresh server per workload (so each table is one depth and one output kind), sends one request (1,200 output tokens,
served sampling), reads the `halo dense profile` tables from `GET /api/logs` (the engine's stderr goes to the server's
log buffer, not the terminal), and stops it. Tables print every 200 windows; the partial table at server stop is not
printed (SIGTERM), so each workload holds 200 or 400 windows. `shares.py` -> `shares.md`; raw tables
`out/*.profile.txt`.

**Caveat:** `--profile` runs the window uncaptured with PDL off, so each stage's ms includes its launch gap. Compare
shares and step changes, not absolute ms. Decode under `--profile` is 3-8% below the served build (smoke: 99.1 against
102.2 t/s at 36K, 71.6 against 77.9 at 137K).

## Stage shares (window-weighted)

| workload | windows | mean T | window ms | attention | attn-core (KV read) | GDN | FFN | LM head | MTP drafter | other |
|---|---|---|---|---|---|---|---|---|---|---|
| short prose | 400 | 3.75 | 30.21 | 7.9% | 1.8% | 20.1% | 46.4% | 3.9% | 20.0% | 1.6% |
| short struct | 200 | 3.88 | 30.72 | 7.8% | 1.7% | 19.8% | 46.6% | 3.8% | 20.7% | 1.3% |
| 36K prose | 400 | 3.35 | 33.36 | 16.3% | 10.6% | 18.0% | 41.1% | 3.5% | 17.8% | 3.4% |
| 36K struct | 200 | 3.21 | 32.31 | 16.6% | 10.7% | 18.4% | 42.4% | 3.5% | 17.9% | 1.2% |
| 137K prose | 400 | 2.68 | 39.02 | 31.4% | 26.7% | 15.0% | 34.1% | 2.9% | 15.0% | 1.7% |
| 137K struct | 200 | 3.36 | 43.23 | 33.4% | 29.1% | 13.8% | 32.2% | 2.7% | 17.0% | 0.9% |

## What it says

1. **attn-core is a step function of ceil(6T/16)**, the S1 diagnosis made directly visible. At 137K: 7.65 ms for
   T = 1-2, 14.9 ms for T = 3-5, 22.3 ms for T = 6-8 (36K: 2.3 / 3.9 / 5.6). `decode_mma_kernel` packs 16 (row, head)
   pairs a block and Qwen3.8-27B has 6 query heads a KV head, so a window of T rows reads the KV ceil(6T/16) times.
   Reading it once (S1) would save 7.2 ms on every T = 3-5 window and 14.6 ms on every T = 6-8 window at 137K:
   about 7% of window time on the prose run and 11% on the struct run, before any drafter change. The value grows
   with depth; it is nil at short context.
2. **The MTP drafter is 15-21% of every window at every depth.** Each draft step adds 1.6 ms (short), 1.7 ms (36K),
   2.0 ms (137K); the full 248,320-row LM head alone is 1.1 ms, so the head is most of a draft step. NInfer's
   `--lm-head-draft` is a frequency-ranked 131,072-row draft head plus an id map (`ninfer-v3`
   `src/targets/qwen3_6_27b/impl/variant.h:50`, `tools/convert/qwen3_6/common/draft_head.py`), the FR-Spec method
   (arXiv 2502.14856). In Halocline it can gather the head's rows as stored (lossless rows), and the verify pass still
   uses the full head, so the output distribution does not change.
3. **GEMV windows above 5 rows step up** (`kMmvqMaxRows = 5`): ffn-up 9.1 -> 10.8 ms and attn-pre 1.16 -> 1.55 ms at
   T = 6; ffn-down climbs at T = 4-5 (5.5, 7.0 ms) and is cheaper again on MMQ at T = 6 (5.7). S2's cost, but mean T
   is 2.7-3.9, so it is the third lever, not the first.
4. GDN recurrence 3% and sampling 0.1%, as the lever map said: not worth work.

## Files

`run.sh` (the lease wrapper), `profile.py`, `shares.py`, `shares.md`, `out/results.json` / `run.log` (per workload tokens,
decode t/s), `out/` (tables, server output). Failed tries kept: `failed-0830/`
(`bin/halocline-vision` not built in the worktree), `try2-0830/` (smoke --long through one server; passed, but the
tables went to the log buffer and were lost at stop).
