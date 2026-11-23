# Speed against NInfer, same weights, same card (2026-11-22)

Halocline (this branch, Q4_K_P pack, rotated 4-bit K and V, 249,856 tokens, `--spec 8`, the 32K draft head) against
NInfer serving HauhauCS's same model with its own served flags. One RTX 4090, both behind the same proxy, streamed,
thinking off, greedy, 1,200 tokens. Two runs each in ABBA order. Decode tok/s as the client sees it.

| Workload | Halocline runs | NInfer runs | Delta |
|---|---|---|---|
| short prose | 90.9, 91.2 | 91.4, 91.1 | -0.2% |
| short JSON | 154.6, 156.4 | 155.4, 154.5 | +0.4% |
| 36K prose | 79.4, 79.2 | 82.6, 82.2 | -3.8% |
| 36K JSON | 145.1, 144.3 | 134.8, 134.1 | +7.6% |
| 137K prose | 67.0, 66.9 | 72.8, 72.5 | -7.8% |
| 137K JSON | 130.9, 130.7 | 121.5, 122.0 | +7.4% |

Tool calls, the 24-case set: 24 of 24 on both, and Halocline agrees with NInfer on 24 of 24
(`tools-score-*.txt`). Every row of `speed.json` holds the first 80 characters of the reply and its hash.
