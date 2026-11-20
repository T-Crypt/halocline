# Halocline image encoder: `--vision cpu` against `--vision dynamic` (2026-11-20)

Question from the airbench Benchmark v1.0 runs ([sparse, 49/49](https://airbench.ai/checkup/64baefc1-476e-41e8-ab11-938f248f8389/report), [dense, 48/49](https://airbench.ai/checkup/02db2264-4211-4dc1-93ba-da42b616e7cc/report)): the sparse run (Qwen3.8-Flash-Next UD-IQ4_XS) waited about 340 s
of its 15m41s vision section for the image encoder. Turns that carried a new image had a median time to first token of
10.6 s, against 0.7 s for text turns (the harness's per-turn timings). Both Halocline entries ran
`--vision cpu`. Is `--vision dynamic` (the encoder inside the engine, on the 4090, no VRAM at rest) faster, and is it
lossless against what we serve?

The 19 images and prompts are the airbench vision set the dense run downloaded on 2026-11-19. They stay outside the
repo, and so do the replies: airbench hides answers on shared reports. The `*.timing.json` files
keep each reply as `reply_sha12` (the first 12 hex digits of its sha256) and `reply_len`.

## Files

| File | What |
|---|---|
| `run.sh` | sparse ABBA under `berth train`: release aa156951, the served sparse flags on :8096, only `--vision` differs |
| `run-dense.sh` | the same for dense: release b2f81178, the served dense flags on :8097 |
| `vision_timing.py` | one fresh request per image (greedy, thinking off, streamed, 64 tokens): time to first token, reply hash; then two 400-token text probes |
| `parity.sh`, `parity.txt` | Halocline's own `vision_cuda_parity encoder` (CUDA encoder against the CPU encoder, per tap) on the 19 images |
| `three_way.cu`, `build.sh`, `three_way.txt` | the CPU encoder at fp32 activations as the reference, against the served CPU path (bf16 GEMM inputs) and the CUDA encoder |
| `N-ARM.*` | per arm (dense arms prefixed `dense-`): `check.*` (the three known-answer images from [`halocline-vision`](../../2026-11-19/halocline-vision/)), `timing.*`, `smoke.*` (the lab's `lsw.py smoke --long`: speed, sampling, tool call, 36K needle, 137K long) |

## Sparse results (`run.log`)

| Arm | Image TTFT, 19 images | Median | Known-answer check | 36K needle prompt / decode | 137K prompt / decode | Text decode after images |
|---|---|---|---|---|---|---|
| 1 cpu | 317.0 s | 16.1 s | 3/3 | 6,055 / 120.3 t/s | 4,751 / 126.3 t/s | 122.3, 133.0 t/s |
| 2 dynamic | **22.3 s** | **1.22 s** | 3/3 | 6,302 / 124.7 | 4,902 / 117.1 | 120.4, 133.6 |
| 3 dynamic | **22.2 s** | **1.22 s** | 3/3 | 6,470 / 114.6 | 5,828 / 116.4 | 116.2, 140.0 |
| 4 cpu | 305.9 s | 15.1 s | 3/3 | 6,483 / 124.8 | 6,026 / 117.7 | 118.2, 136.6 |

Peak VRAM is the same in both modes (23,644 MiB cpu, 23,640 dynamic): the encoder borrows expert-cache slots while it
reads an image and gives them back. Depth prompt and decode speed show no change beyond the spread between equal arms.

**Replies.** Arm 3 (dynamic) and arm 4 (cpu) gave identical replies on 19 of 19 images. Arms 1 and 2 differ from
those on three items, and the repeat of each mode differs as much as the two modes do: cpu against cpu 17/19,
dynamic against dynamic 17/19. One flipped item answered the same wrong way on a cpu arm and a dynamic arm. The flips
come from run-to-run nondeterminism in the language model (arm 1 was the first after a cold load), not from the
encoder. A first reading of arms 1 and 2 alone blamed the GPU encoder; arms 3 and 4 disproved it.

## Encoder numerics (`parity.txt`, `three_way.txt`)

The served CPU sidecar (`tools/vision/vision.cpp`) initialises `CpuEncoder` with `round_bf16 = true`: it rounds every
GEMM input to bf16, as the CUDA encoder does. The two paths differ only in attention (CUDA: fp16 tensor cores, fp32
softmax state; CPU: fp32) and accumulation order. Halocline's parity gate on the CUDA encoder passes on all 19 images
(final embeddings against the CPU: rel L2 mean 8.2e-3, gate 1e-1). The CUDA encode takes 5.8-77 ms per image.

Against the CPU encoder at fp32 activations (the closest thing to exact we have):

| Path | rel L2, mean (max) | Worst row cosine, median (worst) |
|---|---|---|
| served CPU (`--vision cpu`, bf16 GEMM inputs) | 1.300e-2 (2.147e-2) | 0.9909 (0.8804) |
| CUDA (`--vision dynamic`) | 1.275e-2 (2.196e-2) | 0.9911 (0.9495) |

The CUDA encoder is as close to fp32 as the served CPU path: closer on 8 of 19 images, worse on 11, by a margin far
smaller than both paths' distance to fp32. Both carry the same bf16 rounding drift, which is the model's own precision
(BF16 weights). fp16 attention adds nothing measurable. For scale, `tests/vision/cpu_parity.cpp` records that the
reference implementation's own runs with and without flash attention differ by 1e-1 at the output on a photo.

## Dense results (`run-dense.log`)

| Arm | Image TTFT, 19 images | Median | Known-answer check | 36K needle prompt / decode | 137K prompt / decode |
|---|---|---|---|---|---|
| 1 cpu | 305.3 s | 14.8 s | 3/3 | 2,933 / 97.6 t/s | 1,930 / 78.5 t/s |
| 2 dynamic | **19 of 19 refused** | | 1/3 | 2,933 / 98.0 | 1,930 / 80.4 |
| 3 dynamic | **19 of 19 refused** | | 1/3 | 2,931 / 101.3 | 1,929 / 79.9 |
| 4 cpu | 306.5 s | 15.1 s | 3/3 | 2,934 / 101.0 | 1,930 / 80.2 |

The two cpu arms gave identical replies on 19 of 19. With `--vision dynamic` the dense engine loads and serves text
at full speed, but every airbench image fails with HTTP 400:
`qwen35: no device memory for the image encoder (958-1165 MiB)` (the server log). Only the smallest
known-answer image (279 prompt tokens) fit. The dense path's dynamic mode `cudaMalloc`s the encoder arena when an image
arrives (`src/dense/model.cpp`, `Model::vision_attach`); at 262,144 tokens with rotated q4_0 KV the planner has already
given the card away (peak 23,456 MiB of 24,564), and dense has no expert cache to lend, which is where sparse's
dynamic mode borrows from.

A dense fix is an engine design change: lend the encoder memory that is idle while an image is encoded (the prompt
workspace `pre_arena` is the candidate, since an ENC command runs alone on the engine thread and the rows go to the
host), or have the planner keep the arena's size free when vision is on. With the MTP drafter, `pre_arena` holds
`hn = (T + 1) * H`, a row carried between chunks; whether anything in it must survive between commands has to be read
before lending it. Not done here: it needs the gates (greedy identity on image requests, the GPU suite).
The sidecar's `--threads` (default half the 32 hardware threads) is a smaller lever the server does not expose.

## Verdict

- **Sparse:** `--vision dynamic` is lossless against today's served `--vision cpu` by every measure here (replies,
  parity at each tap, distance to fp32), and 14 times faster to the first token on an image. No encoder code change is
  needed. The served sparse entry switches to `--vision dynamic` (2026-11-20).
- **Dense:** stays on `--vision cpu` (about 15 s an image). `--vision dynamic` does not fit beside the 262,144-token
  context; the engine change above is the way to it.
