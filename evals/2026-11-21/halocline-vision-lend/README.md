# The dense image encoder in the borrowed prompt workspace (2026-11-21, ticket 5KWJVRF)

**Why.** Through llama-swap, Halocline dense (Qwen3.8-27B UD-Q4_K_XL, 262,144 tokens) served images on the CPU encoder:
each new screenshot added 13-23 s before the first byte (live captures, 2026-11-21 morning). `--vision dynamic` could
not help: its own arena (958-1,165 MiB) did not fit beside the plan (702 MiB free). OrcaSAQ-2 27B on the same path
logged 15 pictures sent to the CPU (`no device memory for the image encoder (1101 MiB)`).

**Change.** Branch `vision-pre-arena` (public main + halo-cyber's CPU fallback, code only): the encoder borrows
`pre_arena`, the prompt chunks' workspace (1,292 MiB on the 27B), which the planner already counts and which is idle
while an encode runs alone on the engine thread. The token iota is written again on hand-back; the MTP row a chunk
carries goes through the session tail, so nothing else survives in it. Fallbacks: an arena of its own, then the CPU.
Second change (operator: "make 32 threads the default for dense"): a dense model's `--vision cpu` sidecar and the
in-engine CPU fallback use every hardware thread. `orca-vision-lend` is the same two changes on halo-cyber 0fe8224d.

## Results

**ABBA** (`run.sh`, release b2f81178 `--vision cpu` against the branch with `--vision dynamic`, the served dense flags):

| arm | 19 airbench images, TTFT sum / median | known answers | smoke (needle 36K, long 137K) | VRAM peak |
|---|---|---|---|---|
| 1 cpu | 298.1 s / 14.39 s | 3/3, 1.9-2.5 s | PASS; 137K decode 82.4 t/s | 23,456 MiB |
| 2 lend | 10.2 s / 0.53 s | 3/3, 0.3-0.4 s | PASS; 137K decode 82.3 t/s | 23,458 MiB |
| 3 lend | 10.2 s / 0.532 s | 3/3 | PASS; 137K decode 80.4 t/s | 23,458 MiB |
| 4 cpu | 289.7 s / 14.22 s | 3/3 | PASS; 137K decode 82.3 t/s | 23,456 MiB |

Replies (kept as `reply_sha12`; airbench answers stay out of the repo): cpu against cpu 19/19, lend against lend 19/19,
cpu against lend 17/19. The two that differ are `chart-simple` and `chart-complex` in every cross pair: both encoders
start the same method and word it differently inside the 64-token cap; neither reaches its final number. This is the
GPU encoder's rounding (fp16 attention), the encoder sparse has served on since 2026-11-20, not the lend.

**Thread sweep** (`threads.py`, the CPU sidecar alone, one warm-up then 6 images): 8 threads 18.28 s median an image,
16 (the old default) 13.05 s, 24 11.37 s, 32 9.87 s.

**Gates** (`gates.sh`, branch with the thread default): `tools/verify.sh build-89` all PASS but `convert_tests` in the
cpu ctest suite (build-89 was configured with a Python without numpy and regex, which verify.sh flags in its NOTE line;
the converter tests PASS under the venv Python in the same run); GPU suite 125/125 with `HALO_QWEN35_PACK`; the served
`--vision cpu` sidecar runs `--threads 32`, known answers 3/3.

**After a 137K prompt, then squeezed** (`after-long.sh`, both packs on their served profiles, branch builds): smoke
PASS, then the three known answers, then again with `hog.py` holding all but 198 MiB of the card.

| pack | after 137K | 198 MiB free | CPU fallbacks | engine peak before the hog |
|---|---|---|---|---|
| dense UD-Q4_K_XL | 3/3, 0.3-0.4 s | 3/3, 0.2-0.3 s | 0 | 23,470 MiB |
| OrcaSAQ-2 27B (K8V4) | 3/3, 0.2-0.3 s | 3/3, 0.2 s | 0 | 23,198 MiB |

Replies equal the dense lend arm's and Orca's 2026-11-20 `--vision cpu` reference. Engine log: `image encoder: in the
prompt workspace (1292 MiB, the picture needs 966 MiB)`. The `vram peak 23965` in `after-long.log.txt` includes the hog's
own context and hold; the mem logs show the step when it starts.

## Files

`run.sh` / `run.log.txt`, `threads.py` / `threads.txt`, `gates.sh` / `gates.log.txt`, `after-long.sh` / `after-long.log.txt`,
`next.sh` / `pipeline.log.txt` (the unattended order), `hog.py` (test helper, frees on exit), `vision_timing.py` (copied
from 2026-11-20). Per arm: `*.check.*`, `*.timing.*`, `*.smoke.*`, `*.serve.out`, `*.mem.log.txt`. The `run.log.txt` fallback
count reads a wrong log path and always prints 0; the counts above come from the engine logs.

In this copy the logs end in `.log.txt` (the repo ignores `*.log` outside `logs/`).
