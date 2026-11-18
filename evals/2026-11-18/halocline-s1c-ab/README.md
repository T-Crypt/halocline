# S1 by W17: a staged K8V4 decode kernel, 8-12 warps, every window of up to 8 rows (Halocline roadmap D5)

2026-11-18 11:29 to 11:53 MST, aphotic, RTX 4090. Halocline branch `s1c-decode-kernel` (2e277290, from W14's window
kernel), against release `b22ecbb2`, under `berth train --as halocline-s1c-gate` (`gate.sh`, `gate.log`).

## Gates

- GPU suite 124/124 (`ctest-gpu.log`). `gqa_k8v4_parity` run directly (`../halocline-decode-kld/w17-parity.txt`):
  "1-9 rows against the 16-row-tile launch: bitwise yes" on its 2,200-cell cache.
- Depth KLD (the maintainer's gate for a decode-kernel change): `../halocline-decode-kld` (`s1c-k8v4-*`, `paired.txt`).

## Kernel bench (`bench.txt`, `../halocline-s1b-ab/bench-s1c.txt`), ms per launch, one K8V4 layer

| cells | T | W17 kernel | 16-row tiles (release) |
|---|---|---|---|
| 300 | 1 / 3 | 0.013 / 0.022 | 0.016 / 0.026 |
| 4,096 | 1 / 3 | 0.026 / 0.037 | 0.025 / 0.046 |
| 36,471 | 1 / 3 / 6 / 8 | 0.100 / 0.128 / 0.148 / 0.160 | 0.108 / 0.207 / 0.311 / 0.323 |
| 137,404 | 1 / 3 / 6 / 8 | 0.342 / 0.427 / 0.464 / 0.478 | 0.473 / 0.926 / 1.383 / 1.390 |

One pass at 137K: 655 GB/s at T = 1 (474 before), 469 GB/s at T = 8 (161 before).

## A/B, ABBA rel / s1c / s1c / rel (`ab.log`, `ab/results.json`), greedy, 1,200 tokens, decode t/s

| workload | release b22ecbb2 | W17 |
|---|---|---|
| short prose | 198.2, 198.1 | 199.0, 199.0 |
| short struct | 196.1, 196.0 | 196.9, 196.8 |
| 36K prose | 150.6, 150.6 | 158.8, 158.9 |
| 36K struct | 170.1, 170.0 | 182.1, 182.1 |
| 137K prose | 102.9, 102.7 | 145.2, 145.2 |
| 137K struct | 115.0, 115.0 | 153.5, 153.4 |

**Caveat: the arms did not emit the same text at depth.** `text_sha` per reply: identical across all four arms at short
prose, short struct and 36K prose; at 137K each build repeats its own text and the two builds differ; at 36K struct the
release differs from itself. Cause: the draft-width policy is driven by measured round times
(`src/spec/draft_policy.cpp`, `cost_ms`), so window widths follow timing, and a window past 5 rows takes MMQ instead of
MMVQ (different rounding); a faster attention kernel makes wide windows cheaper and the policy takes them. The 137K
numbers are therefore end to end, kernel plus the wider windows it makes worth taking, on texts that differ. The quality
question is the depth-KLD gate's, which scores fixed 3-row windows (`--quality-t` default) and is deterministic.
