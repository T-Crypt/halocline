# Dense KLD at depth: int8, K8V4 and rotated 4-bit KV (roadmap D8 item 3)

2026-11-17 20:20 to 2026-11-18 00:49, aphotic, RTX 4090 at 350 W, driver 615.78.08. Qwen3.8-27B UD-Q4_K_XL
(`Qwen3.8-27B-UD-Q4_K_XL.halo`), `--spec 8 --mtp-kv int8`, `halocline-bench --quality 16` against BF16 depth bases:
16 chunks, each the first DEPTH tokens of wikitext-103 train as a prefix, then 1,024 scored rows
(`tools/depth_kld`, llama.cpp 2cbe0ec with `llama-kl-first.patch`, BF16 GGUF). `--max-context` is depth + 1,033: the
engine admits prompt + max_new + 8 <= context. Builds: `d8-integration` (int8, K8V4), `d8-q4` (`--kv q4_0`), and the
drafter ring fix (`mtp-ring-fix` 94ec6c75 / `d8-q4-ring` a9b4a774) for every `--mtp-window` arm. Scripts: `run.sh`
(arms), `window5.sh`, `ringfix.sh`; log `run.log`; paired tables `paired-depth.txt` (`kv_paired.py`, 95% t interval
over the 16 chunks).

| Depth | KV | mean KLD | top-1 % | PPL / base | paired dKLD vs int8 [95% CI] |
|---|---|---|---|---|---|
| 65,536 | int8 | 0.00418 | 96.97 | 7.0469 / 7.0407 | (reference) |
| 65,536 | K8V4 | 0.00490 | 96.75 | 7.0531 / 7.0407 | +0.00072 [+0.00050, +0.00095] |
| 65,536 | rotated 4-bit K and V | 0.00602 | 96.58 | 7.0502 / 7.0407 | +0.00185 [+0.00135, +0.00234] |
| 140,288 | int8 (`--mtp-window 32768`, fixed ring) | 0.00494 | 97.02 | 5.8398 / 5.8356 | (reference) |
| 140,288 | K8V4 | 0.00560 | 96.80 | 5.8471 / 5.8356 | +0.00066 [+0.00050, +0.00082] |
| 174,080 | K8V4 | 0.00654 | 96.62 | 6.0947 / 6.0722 | no int8 arm (int8 does not fit) |
| 261,119 | rotated 4-bit K and V | not measured | | | see below |

Read: K8V4 costs the same over int8 at 65K and 140K (+0.0007), as at 36K (+0.00093, `halocline-d8-kv`). Rotated
4-bit costs 2.6 times K8V4's increment at 65K (+0.00185 against +0.00072). KLD levels differ by depth because each
depth scores different text; compare formats within a depth only.

## Failures, kept

- `int8-140288` (21:04): `chunk 0: ERR qwen35: the drafter failed`. Cause: `gqa_decode_attn` swept a ring from chunk 0,
  so past position `cap` a window read no cell and the drafter's logits came out NaN. At 36K the window was only partly
  outside the sweep, so `qwen35_mtp_cache_test` passed with wrong drafts and an unchanged output. Fixed on branch
  `mtp-ring-fix` (94ec6c75): the sweep starts at the window's first chunk. `gqa_attn_parity`'s new ring case (a 1,015-cell
  ring three laps in against the same window from an absolute cache, 66 windows of 1, 7 and 16 rows) reads max abs 0,
  all finite (`ring-parity-*.log`); `qwen35_mtp_cache_test` all ok; the 2-chunk repro and the full arm ran
  (`int8-140288-ring-repro`, `int8-140288-ring`).
- `q4-261119` and `q4-261119-ring`: the engine refuses `--max-context 262152` (beyond the model's 262,144). The 261,119
  base scores rows up to cell 262,143, and the engine wants 8 cells of headroom past the last generated token, so this
  protocol cannot run at that depth without YaRN. Open: score the rows that fit (1,016 of 1,024 per chunk) or build a
  base at 261,111.
