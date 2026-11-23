# What fits on 24 GB (2026-11-22)

The Q4_K_P pack with the projector, `--vision dynamic`, rotated 4-bit K and V, int8 MTP cache, one RTX 4090 with
nothing else on it. The planner keeps `main`'s 512 MiB margin.

| `--max-context` | Planner | Why |
|---|---|---|
| 262,144 | refused | the session needs 6,902 MiB, 7,199 MiB free: 215 MiB short after the margin |
| 253,952 | refused | needs 6,758 MiB: 71 MiB short |
| 249,856 | fits | served |

At 249,856: a needle placed in a 200,887-token prompt came back right (prefill 127 s), the three known-answer images
gave 3 of 3 after that fill, and the card peaked at 23,478 MiB.

int8 K and V fit about 128K on this pack, so the served profile uses the rotated 4-bit cache. Its cost is in
[`../gate/results.md`](../gate/results.md): at 65,536 tokens, +0.0023 dNLL over int8.
