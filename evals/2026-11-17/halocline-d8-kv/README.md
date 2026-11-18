# Halocline dense: K8V4 and rotated 4-bit KV against int8 (D8 items 1 and 3), 2026-11-17 (the workstation, RTX 4090, the board)

Qwen3.8-27B UD-Q4_K_XL (Unsloth v3.1, the pack of `../halocline-dense-quality/`), the D4 protocol of that directory:
`halocline-bench -t conv,struct,prose -p 16384,36864 --seed 1 --quality 100 --kld-base qwen38-27b-bf16-base.kld --
--spec 8 --max-context 37528 --kv ...` (100 chunks of 512 against BF16, 25,500 scored rows). One engine on the card,
each window under a Berth training lease; 350 W enforced limit, driver 615.78.08. Runners: `gpu-window.sh` (window 1:
the three quality arms), `window2.sh` (window 2: the fixed tests, then prefill ABBA). Paired intervals: `kv_paired.py`
(`paired.md`).

Builds: arm `int8` started from `b7053a15`, arms `k8v4` and `k8v4-mtpint8` from `07f11e67` (the K8V4 prompt sign fix
`2b6095a4`; the int8 path is identical in both). Arm `q4` (`--kv q4_0 --mtp-kv int8`, rotated q4_0 K and V) from branch
`d8-q4` at its merge with `d8-integration` (`63eb5db7`); its `bench.json` names `add8d0ef` because the build id is
taken when CMake configures, and that tree was configured before the merge and rebuilt after it. Prefill ABBA (`pf-*`) from `7e0f43c4` (the K8V4 prompt kernel at the
int8 planner's 8-warp shapes).

## Quality (100 chunks)

| KV | bytes a cell a layer | mean KLD | top-1 | PPL (base 6.7411) | paired dKLD vs int8 [95% CI] |
|---|---|---|---|---|---|
| int8 | 2,112 | **0.00582** | **96.67%** | 6.7336 | reference |
| k8v4 | 1,632 | 0.00676 | 96.39% | 6.7412 | +0.00093 [+0.00074, +0.00113], top-1 -0.28 points |
| k8v4, `--mtp-kv int8` | 1,632 | 0.00676 | 96.39% | 6.7412 | +0.00093 [+0.00074, +0.00113] |
| q4 (`--kv q4_0`), `--mtp-kv int8` | 1,152 | 0.00779 | 96.07% | 6.7482 | +0.00197 [+0.00167, +0.00228], top-1 -0.60 points; +0.00104 [+0.00082, +0.00125] over k8v4 |

- int8 reproduces the D4 number exactly (0.005822, `../halocline-dense-quality/q4kxl-int8-mmvq/`).
- **K8V4 costs +0.00093 mean KLD** (16% over int8) and 0.28 top-1 points, measurably, at 36K context.
- **The drafter's KV format does not touch the output**: `k8v4` and `k8v4-mtpint8` have byte-identical `rows.tsv`.
  The MTP drafter only proposes; the verify pass decides every emitted token.
- **Rotated 4-bit K costs about as much again as 4-bit V**: q4 sits +0.00104 above k8v4, which sits +0.00093 above
  int8. Each 4-bit half adds roughly 0.001 mean KLD at 36K.
- This gate scores 512-token chunks. KV rounding that builds up deep in a context is the depth gate's job
  (`../halocline-d8-depth/`, BF16 bases at 65,536 / 140,288 / 174,080 tokens).

## Speed

Decode (one repetition, sampled, `--spec 8`, MTP; window 1 builds):

| KV | conv | struct | prose | accepted conv / struct / prose |
|---|---|---|---|---|
| int8 | 98.5 | 192.7 | 86.8 | 66.6 / 96.8 / 60.9% |
| k8v4 | 100.6 | 192.3 | 85.5 | 68.6 / 96.9 / 59.0% |
| k8v4, `--mtp-kv int8` | 98.4 | 192.5 | 85.4 | 65.8 / 96.7 / 60.0% |
| q4, `--mtp-kv int8` | 100.7 | 191.1 | 86.6 | 66.6 / 96.6 / 61.6% |

Decode is within 2% across the three: at 36K the KV is a small share of a window's bytes. The drafter's int8 cache
changes the drafts (acceptance moves by up to 3 points per fixture) but not the speed beyond one repetition's noise.

Prefill, engine tokens/s. Window 1 (4-warp K8V4 kernel, one repetition each) and window 2 (8-warp kernel, ABBA:
`pf-int8-a`, `pf-k8v4-a`, `pf-k8v4-b`, `pf-int8-b`, conv decode as the smallest fixture):

| KV | 16K | 36K |
|---|---|---|
| int8 (window 1) | 3,424 | 3,242 |
| k8v4, 4-warp (window 1) | 3,195 (-6.7%) | 2,831 (-12.7%) |
| int8, ABBA mean (window 2) | 3,419 | 3,238 |
| k8v4, 8-warp, ABBA mean (window 2) | 3,346 (-2.1%) | 3,090 (-4.6%) |
| q4, 4-warp (one repetition) | 3,151 (-8.0%) | 2,745 (-15.3%) |

The ABBA pairs agree to 0.2%. q4's prompt kernel runs 4 warps only: its 8-warp shape would need 102,400 B of shared
memory, 1 KiB over sm_89's opt-in maximum, and expanding K from q4_0 blocks in the tile load adds work K8V4 does not have. The 8-warp shape takes 101,376 B of shared memory, exactly sm_89's per-block opt-in
maximum. The remaining 4.6% at 36K has two suspects, neither measured yet: the in-write Hadamard transform (17 barriers
a row in `attn_qk_prepare`) and the synchronous q4 V expansion in the tile load.

## Context (the planner, `tests/kernels/dense_kv_planner_test.cpp`)

7,597 MiB free after the UD-Q4_K_XL vision pack's weights, plus the 512 MiB reserve; MTP and vision on:

| Drafter KV | int8 | k8v4 | q4 |
|---|---|---|---|
| fp16, full context | 130,832 | 164,094 | 220,035 |
| int8, full context (`--mtp-kv int8`) | 138,061 | **175,629** | 241,285 |
| fp16, 32K ring (`--mtp-window 32768`) | 142,716 | 184,692 | 261,647 |
| int8, 32K ring | 144,641 | 187,183 | **265,176** |

`--kv k8v4 --mtp-kv int8` is the configuration this eval recommends for long context: 175,629 cells (43% more than
the served 122,880), the same output as `--kv k8v4`, and it keeps parking and the conversation cache, which the ring
gives up. 262,144 cells need 4-bit K as well (roadmap D8 item 3): q4 reaches 241,285 cells with `--mtp-kv int8` and
265,176 adding the 32K ring, so 262,144 fits only with both (53 MiB to spare).

## Tests

`ctest-gpu-b7053a15.txt`: the GPU suite at `b7053a15`, 122 of 124; the two failures were the new tests themselves
(`gqa_k8v4_parity`, whose reference was wrong in three places, and `qwen35_mtp_cache_test`'s drafter-window arm).
`ctest-gpu-fixed-7e0f43c4.txt`: both fixed, plus `gqa_attn_parity`, on the real pack: 3 of 3. On `d8-q4`
(`aa17ca93`), `gqa_k8v4_parity` including its q4 arm (decode 0.00082, prompt 0.0234 against fp64) and
`gqa_attn_parity` pass on the GPU.
