# Halocline dense: D4 quality gate, D5 speed, first D7 arm (llama.cpp), 2026-11-17 (the workstation, RTX 4090, the board)

Qwen3.8-27B from Unsloth's v3.1 GGUFs (revision `4af51b35`), repacked losslessly (D2, `ffb25a21`), against the frozen
BF16 base `/data/models/halocline/kld-bases/qwen38-27b-bf16-base.kld` (sha256 `9ffb3bd9…`, wikitext-2, 100 chunks of
512, 25,500 scored rows). One engine on the card at a time, each under a Berth foreground lease; 350 W enforced limit,
driver 615.78.08. Build: `~/wt/halocline-import` at `ffb25a21` plus the uncommitted dense engine change
(`../halocline-dense-load/README.md`).

Runners: `d4.sh` (`halo-bench -t conv,struct,prose -p 16384,36864 --seed 1 --quality 100 --kld-base BASE -- --kv int8
--spec 8`, sampled decode at the fixtures' temperature, MTP drafter on) and `llama-kld.sh` (`llama-perplexity
--kl-divergence --chunks 100 -c 512 -b 512 -ub 512 -ngl 99 -fa on`, llama.cpp `2cbe0ec`). Paired intervals:
`kv_paired.py`.

## Quality (100 chunks)

| Arm | Weights | KV | Mean KLD | Median | p99 | Top-1 | PPL (base 6.7411) |
|---|---|---|---|---|---|---|---|
| Halocline, decode on MMQ | UD-IQ4_XS | int8 | 0.016176 | 0.00721 | 0.161 | 94.31% | 6.8004 |
| Halocline, decode on mmvq (T <= 8) | UD-IQ4_XS | int8 | 0.016328 | 0.00725 | 0.164 | 94.30% | 6.8013 |
| Halocline, decode on mmvq (T <= 8) | UD-Q4_K_XL | int8 | **0.005822** | 0.00230 | 0.0601 | **96.67%** | 6.7336 |
| llama.cpp `2cbe0ec` | UD-Q4_K_XL | f16 | 0.005730 ± 0.000230 | 0.00225 | 0.0567 | 96.54 ± 0.11% | 6.7299 |
| llama.cpp `2cbe0ec` | UD-IQ4_XS | f16 | 0.016136 ± 0.000377 | 0.00717 | 0.157 | 94.34 ± 0.15% | 6.7990 |
| llama.cpp `2cbe0ec` | UD-IQ4_XS | q8_0 | 0.016384 ± 0.000398 | 0.00715 | 0.159 | 94.35 ± 0.15% | 6.8013 |

- **UD-Q4_K_XL is 2.8 times better than UD-IQ4_XS on mean KLD** (0.0058 vs 0.0162) and 2.4 points better on top-1.
- **Halocline at 8-bit KV beats llama.cpp at 8-bit KV** on IQ4_XS (0.016176 vs 0.016384) and is within error of
  llama.cpp's f16 KV (0.016136).
- **Halocline matches llama.cpp on the same weights:** mean KLD within llama.cpp's own error bar (+0.00009 against
  ±0.00023), top-1 0.13 points higher, with an int8 KV cache against llama.cpp's f16.
- **mmvq vs MMQ for decode windows** (paired, IQ4_XS): mmvq costs +0.00015 mean KLD [95% CI +0.00005, +0.00025] and
  -0.02 top-1 points. Likely cause: mmvq's q8_1 keeps block scale and sum in fp16, MMQ's tile layout in fp32. Small,
  real, and the price of +18-28% decode. mmvq is the default (decode without it is below the NInfer floor);
  `HALO_DENSE_MMVQ=0` is the MMQ arm. A decode GEMV with fp32 activation scales is the plan to close the gap
  (docs/roadmap.md (D8, Q1-Q4)).

## Speed (one repetition, sampled, `--spec 8`, MTP)

| Arm | conv | struct | prose | prefill 16K | prefill 36K |
|---|---|---|---|---|---|
| IQ4_XS, MMQ | 96.3 | 197.1 | 77.0 | 3,583 | 3,383 |
| IQ4_XS, mmvq T <= 8 | 113.3 | 178.0 | 98.2 | 3,574 | 3,373 |
| Q4_K_XL, mmvq T <= 8 | 98.1 | 153.8 | 86.5 | 3,417 | 3,234 |
| NInfer floor (2026-11-16, `lsw.py smoke`) | 112-118 warm decode | | | | 3,215 |

Verify time per window (draft logs, IQ4_XS, MMQ / mmvq): T1 23.8 / 18.2 ms, T3 23.4 / 18.1, T5 24.1 / 21.3, T6 24.4 / 25.6,
T8 25.0 / 32.8. The crossover is between 5 and 6 rows, so the build now caps mmvq at 5 rows (`kMmvqMaxRows`); the
rows above are from the uncapped build (struct, with mostly 8-row windows, is understated for mmvq). Prefill beats the
NInfer 36K floor on both quants. Decode fixtures here are not the NInfer smoke: the like-for-like decode comparison is
the same `lsw.py smoke --long` through llama-swap, still to run.

## 262K fit (probe, vision packs, `--spec 8`, `queue3.sh`, `fit/`)

| Pack | Free after the weights | 262K at int8 KV needs | at fp16 KV |
|---|---|---|---|
| UD-IQ4_XS-vision | 10,589 MiB | 11,830 MiB (1.2 GiB short) | 19,766 |
| UD-Q4_K_XL-vision | 7,597 MiB | 11,830 MiB (4.2 GiB short) | 19,766 |

Of the 11,830 MiB: 8,448 int8 KV, 1,024 the MTP drafter's fp16 KV at full context, the rest GDN state and workspace.
Neither quant reaches 262K with the dense path's KV formats today; K8V4 and 4-bit KV in the dense kernels are the
next engine job (roadmap D8).

## Open from this run

- llama.cpp at `-ub 8192` (how the base was made) runs out of memory on 24 GB: an 8.4 GiB compute buffer for the 248K
  vocabulary. The arms here use `-ub 512`.
- Long-context KLD at depth, NInfer's KLD, exact Q8_0 GDN gates, FP32 residual, upstream sync: the maintainer notes
  and the repo's docs/roadmap.md (D8, Q1-Q4).
