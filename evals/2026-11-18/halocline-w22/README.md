# W22: S1's staged decode kernel for the rotated 4-bit cache (2026-11-18)

Halocline branch `q4-decode-kernel` (cc96bc8d): `decode_k8v4_kernel<..., Q4>` stages K's rotated q4_0 blocks (9,216 B a
chunk against K8V4's 13,056) and dequantizes them with the tiled Q4 kernel's statements. Under `berth train --as
halocline-w22-gate` (`gate.sh`, `run.log`).

- GPU suite 124/124 (`ctest-gpu.log`); `gqa_k8v4_parity` (`parity.txt`): the engine's Q4 launch against the 16-row tiles at
  1-16 rows, **bitwise yes**.
- SASS: every kernel of the served build (long-context, release ea5868de's code) is instruction-identical in this build,
  K8V4's staged kernels included; three Q4 instantiations are new.
- Depth KLD with `--long-context`, paired against the tiled kernel's arms (`../halocline-long-context`): **+0.00000** at
  65,536 and 140,288 (`paired.txt`).

## Speed (`ab-*.json`), --long-context, thinking off, greedy, 1,200 tokens, streamed, client rate, ABBA

| workload | release ea5868de (tiled) | W22 | change |
|---|---|---|---|
| short prose | 81.6, 81.6 | 81.5, 81.4 | even |
| short struct | 152.4, 149.5 | 151.0, 153.4 | even |
| 36K prose | 68.7, 68.7 | 73.9, 74.0 | +7.6% |
| 36K struct | 131.4, 130.0 | 148.8, 145.1 | +12.4% |
| 137K prose | 52.7, 52.8 | 62.7, 64.2 | +20.3% |
| 137K struct | 88.5, 87.4 | 128.8, 129.9 | +47.1% |

Short context is unchanged (the KV read is small there); the gain grows with depth, as S1's did for K8V4.
