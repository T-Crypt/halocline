# halo-uncensored gate: against HauhauCS Q8_K_P (llama.cpp, f16 K/V)

## Short gate: wikitext-2, 100 x 512

| engine | weights | K/V | mean KLD | top-1 % | PPL |
|---|---|---|---|---|---|
| Halocline | Q4_K_P | int8 | 0.008864 | 96.07 | 6.9649 |
| Halocline | Q4_K_P | rotated 4-bit | 0.011277 | 95.38 | 6.9841 |
| llama.cpp | IQ4_XS | q8_0 | 0.019033 | 94.14 | 6.9849 |

## Depth 65,536 (16 chunks x 1,024 rows; base PPL 7.0763)

| engine | weights | K/V | mean KLD | top-1 % | dNLL vs Q8_K_P [95% CI] | vs Halocline Q4_K_P 4-bit, paired |
|---|---|---|---|---|---|---|
| Halocline | Q4_K_P | rotated 4-bit | 0.007665 | 95.96 | +0.00649 [+0.00424, +0.00875] | (reference) |
| Halocline | Q4_K_P | int8 | 0.006093 | 96.36 | +0.00417 [+0.00215, +0.00619] | -0.00233 [-0.00389, -0.00076] |
| llama.cpp | IQ4_XS | q8_0 | 0.013111 | 94.74 | | |
| NInfer (HauhauCS weights) | its v3 requant | rk4v4-e8 (served) | no logits | | +0.02237 [+0.01876, +0.02598] | +0.01587 [+0.01153, +0.02021] |
| NInfer (HauhauCS weights) | its v3 requant | int8 | no logits | | +0.01895 [+0.01599, +0.02190] | +0.01245 [+0.00862, +0.01629] |

## Depth 140,288 (16 chunks x 1,024 rows; base PPL 5.8657)

| engine | weights | K/V | mean KLD | top-1 % | dNLL vs Q8_K_P [95% CI] | vs Halocline Q4_K_P 4-bit, paired |
|---|---|---|---|---|---|---|
| Halocline | Q4_K_P | rotated 4-bit | 0.009663 | 95.82 | +0.00696 [+0.00432, +0.00960] | (reference) |
| NInfer (HauhauCS weights) | its v3 requant | rk4v4-e8 (served) | no logits | | +0.02037 [+0.01359, +0.02714] | +0.01341 [+0.00678, +0.02004] |

## Method

- Reference: HauhauCS's Q8_K_P in llama.cpp, f16 K and V, `--kl-divergence-base`. The depth bases keep only the last
  1,024 positions of each chunk.
- Halocline: `halocline-bench --quality` on the same tokens, against the same base files.
- NInfer exposes no logits, so it gets dNLL on the target tokens of the same streams, and a paired interval against
  Halocline's served profile.
- Served profile: rotated 4-bit K and V (`--kv q4_0`), int8 MTP cache.
