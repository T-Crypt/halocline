# Halocline dense 262K profile (--long-context): gate, served test, switch (2026-11-18)

Branch `long-context` (d8-q4's rotated q4_0 K and V merged onto main with S1, plus `--long-context`), release
`~/halocline/releases/ea5868de`. Operator decision: the 262K profile is the dense default (Halocline roadmap standing
order 8). Under `berth train`.

## Gate (`gate.sh`, `run.log`, `paired.txt`)

- GPU suite 124/124 (`ctest-gpu.log`). An earlier start is in `stopped-1/` (stopped to make the Q4 cache write its own
  instantiation; see SASS below).
- Default cache unchanged: K8V4 depth KLD paired against the 2026-11-17 reference, **+0.00000** at 65,536 and 140,288.
- SASS (`sass-identity.txt`): fp16/int8/K8V4 instantiations of attn_qk_prepare and the prompt kernels identical to release
  75726a0f; decode kernels identical except constant-bank offsets.

| depth | K8V4 mean KLD | 262K (`--long-context`) mean KLD | paired vs K8V4 [95% CI] | top-1 points |
|---|---|---|---|---|
| 65,536 | 0.00490 | 0.00602 | +0.00112 [+0.00080, +0.00144] | -0.17 |
| 140,288 | 0.00560 | 0.00676 | +0.00116 [+0.00096, +0.00137] | -0.15 |
| 174,080 | 0.00654 | 0.00762 | +0.00108 [+0.00092, +0.00125] | -0.33 |

Port check: the merged build's 65,536 arm against d8-q4's own q4-65536: +0.00000. The cost does not grow with depth.

## Served

- From the release as a repo user runs it (`live262.sh`: `bin/halocline serve --config examples/qwen38-27b.json`):
  ctx 262,144, the lossy warning at start, a needle at **252,483 tokens found** (prompt 1,110 t/s, 227 s). The smoke in
  that run failed on a 404 from the harness (it asked for the profile name, the server's id is the pack's), not the engine.
- llama-swap `Halocline-27B-UD-Q4_K_XL` switched to ea5868de `--long-context` (the lab's llama-swap config), then
  `lsw.py smoke --long` through llama-swap (`served/smoke-long.txt`): all PASS.

| | K8V4 on 75726a0f (served before) | 262K on ea5868de |
|---|---|---|
| warm decode | 113.3-145.9 (thinking on) | 92.3-113.6 |
| 36K needle decode / prompt | 102.9 / ~3,137 | 84.9 / 2,783 |
| 137K needle decode / prompt | 88.0 / ~2,295 | 62.3 / 1,644 |

The speed cost is the attention kernel: a 4-bit cache decodes on the 16-row tiled kernel (S1's staged kernel reads K8V4
only) and prefills on the 4-warp prompt_v4 kernel. The lossless fix is W22's q4-K instantiation of S1's kernel.
