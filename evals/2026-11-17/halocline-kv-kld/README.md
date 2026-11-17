# Halocline: KV format by KLD to BF16, then speed, 2026-11-17 (the workstation, RTX 4090, ticket the board)

Maintainer ask (2026-11-17): "I want the best performance to KLD on the BF16 full precision model, so whatever you
suggest that gets us the closest without a huge performance loss." This picks the KV format, then the drafter window,
for the llama-swap entry.

- Build: release tree `~/halocline/releases/d314a21a` (git archive of `scope-wave1` `d314a21a`, `build-89`; the JSON
  says `commit: unknown` because the archive has no `.git`).
- Pack: `/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt` (90.08 GiB, 15 sections, 343 s): the
  first-load pack's command plus `--mtp-source /data/models/halocline/mtp-qfn` (the BF16 `mtp.*` tensors of
  `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`), so it carries the MTP drafter. hc_* still cast to BF16 (`--compat-bf16`,
  maintainer: keep for now).
- Base: `/data/models/halocline/kld-bases/flashnext-bf16-base.kld` (Unsloth BF16 GGUF through llama-perplexity,
  wikitext-2, 512-token chunks, base PPL 3.2733).
- Runs (`kv-sweep.sh`): per KV, one `basalt-bench` under its own Berth foreground pinned lease, card otherwise idle:
  decode fixtures conv/struct/prose (`--seed 1`, `--spec 8`), prefill 32K and 131K, then the quality gate on **100
  chunks** (25,500 scored rows; the gate adds `--spec 5 --adapt-swaps 0 --resident-experts`, T 3). No lossy flags.
- Paired comparison (`kv_paired.py`, `paired.md`): per-chunk mean KLD difference against int8, mean over 100 chunks with
  a 95% t interval. Raw rows are published as `rows/<kv>.tsv` beside this file.

## Quality (100 chunks, against BF16)

| KV | PPL | mean KLD | top-1 % | paired dKLD vs int8 [95% CI] | top-1 points | verdict |
|---|---|---|---|---|---|---|
| **int8** | 3.2323 | **0.08195** | **90.45** | reference | | |
| fp16 | 3.2320 | 0.08230 | 90.33 | +0.00035 [-0.00061, +0.00131] | -0.13 | same as int8 (interval holds 0) |
| k8v4 | 3.2424 | 0.08621 | 90.26 | +0.00426 [+0.00307, +0.00545] | -0.20 | measurably further from BF16 |
| q4_0 | 3.2399 | 0.08843 | 90.03 | +0.00647 [+0.00499, +0.00795] | -0.43 | measurably further from BF16 |

Matches upstream's published costs on the NVFP4 pack (T3: int8 vs fp16 lossless, k8v4 +0.0056, q4_0 +0.0093). Almost
all of the 0.082 is the UD-IQ4_XS quant itself; the KV format moves it by at most 8%.

## Speed (drafter on, `--spec 8`, one repetition)

| KV | conv tok/s | struct tok/s | prose tok/s | prefill 32K t/s | prefill 131K t/s |
|---|---|---|---|---|---|
| **int8** | 131.8 | **264.6** | **133.7** | **6,053** | **6,019** |
| fp16 | 135.4 | 259.5 | 131.0 | 4,773 | 2,997 |
| k8v4 | 129.7 | 258.3 | 133.2 | 5,421 | 5,694 |
| q4_0 | 127.0 | 238.2 | 127.8 | 5,562 | 5,700 |

conv and prose are sampled (seed 1), so their token counts differ between formats; struct is the cleanest decode
comparison. fp16 KV takes VRAM from the expert cache, which halves 131K prefill.

## Choice: `--kv int8`

Closest to BF16 within noise (tied with fp16, ahead of k8v4 and q4_0) and the fastest prefill at every depth. Compared
with the no-drafter serve (`../halocline-sparse-serve/`): warm decode 103 -> 132-134 tok/s on chat-like text, 265 on
structured output.

Limit: the gate scores 512-token chunks, so it cannot show KV rounding that only builds up deep in a long context.
The long-context needle checks of `lsw.py smoke --long` are the guard there.

## Drafter window (`spec-sweep.sh`, int8, decode fixtures, 2 repetitions, mean)

| `--spec` | conv tok/s | struct tok/s | prose tok/s | acceptance conv / struct / prose % | struct tokens per pass |
|---|---|---|---|---|---|
| 5 | 134.8 | 238.1 | 136.3 | 74.3 / 98.5 / 67.2 | 5.04 |
| **8** | **137.5** | 266.0 | 133.0 | 74.0 / 97.5 / 64.9 | 7.80 |
| 12 | 133.4 | 267.9 | 133.3 | 73.8 / 97.5 / 64.6 | 7.80 |

Window 5 caps structured output (11% slower). Window 12 drafts no further than 8 (7.8 tokens per pass either way) and
gains nothing. **`--spec 8` stays** (the engine default). Run 2026-11-17 08:14-08:27 under a Berth lease that drained
NInfer (maintainer: "go now").
