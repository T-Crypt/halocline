# Roadmap and open work

The engine has two paths. The sparse path runs Qwen3.8-Flash-Next today. The dense path is for Qwen3.8-27B now and
Qwen4-27B later. Since 2026-11-17 it loads Unsloth's UD GGUFs losslessly and runs: at llama.cpp parity on KLD for the
same weights, above the NInfer prefill floor. It does not reach 262K context yet (D8).

The dense goal, set by the maintainer on 2026-11-17: **Qwen3.8-27B on Unsloth's v3.1 GGUF quants only, held to the same
gates as the sparse path, with lower KLD to BF16 and better output quality than NInfer and llama.cpp on the same card.**
Added the same day: **262,144 tokens of context is a hard requirement, with MTP and vision**, the highest-quality UD quant
that fits with it, and no loss of speed against NInfer anywhere. A 4-bit quant only for a measured reason.

## Standing orders (maintainer, 2026-11-17 evening)

These hold for every change until the maintainer replaces them.

1. **The dense baseline is being set now, on Qwen3.8-27B UD-Q4_K_XL.** It is the comparison arm for Qwen4-27B when its
   weights are public (the architecture diff is scoped then, P3). Q4_K_XL is the dense quant; IQ4_XS is not wanted.
   Slightly slower decode than the sparse path is acceptable; quality is not traded.
2. **Dense is measured against NInfer the way sparse was measured against Strata:** one card, one engine at a time,
   the same fixtures (decode conv/struct/prose `--seed 1`, prefill at fixed depths), power limit recorded, ABBA for
   speed, the 100-chunk KLD gate against the same BF16 base, depth KLD (P0 item 3), greedy identity across builds,
   sampling distribution, tool-call breadth. The goal is **large gains over NInfer**, reported with intervals. Where
   Halocline does not win, the README says so.
3. **Both paths keep their baselines.** A change that touches shared code (kernels, KV formats, planner, serve, the
   model file, the CLI) runs both paths' gates before it ships: the sparse gate (greedy identity 900/900 against the
   served release, KLD 0.08195 / top-1 90.45% at int8, decode 131.8 / 264.6 / 133.7 tok/s, prefill 6,053 / 6,019 at
   32K / 131K, `halocline-kv-kld`) and the dense gate (KLD 0.00582 / top-1 96.67% at int8, the D4/D5 fixtures).
   A sparse loss from dense work is a regression, not a trade.
4. **Levers come from research with citations.** A dense lever is proposed with its paper or measured source (arXiv,
   Hugging Face, the lab's research base), A/B'd on the gates above, and kept only on a measured win.
5. **Model targets stay open.** Qwen4-27B and a Qwen4 Flash are not public. Whether Qwen3.8-Flash-Next stays the
   Flash target or a new Flash ships is unknown: the sparse path keeps Flash-Next as its baseline until weights land.

Status:

| Area | State |
|---|---|
| Sparse path (Flash-Next) | runs, gated, served through llama-swap |
| Dense path (27B) | D2-D4 and D6 done on UD-IQ4_XS and UD-Q4_K_XL, vision packs built; 262K does not fit with the KV formats the dense path has (D8) |
| Dense vs llama.cpp | parity on KLD for the same GGUF; better than llama.cpp at the same 8-bit KV precision (D4) |
| Dense vs NInfer | prefill above its floor; decode near it; KLD and the like-for-like smoke not measured (D7) |
| Qwen4-Flash / Qwen4-27B | target only; weights are not public |
| Task-level eval | not started |
| Sampling, long-context KLD, Strata KLD | not measured |

## P1: the dense path (Qwen3.8-27B), the priority

The model is a dense hybrid: 64 layers, hidden size 5,120, three Gated DeltaNet layers to one gated attention layer,
SwiGLU FFN in each, 248,320 vocabulary, 262,144 native context. No experts, no router, no expert tiering. That is why
it is the quality path: there is less machinery between the weights and the logits than the sparse model has.

What exists: `src/models/qwen35/adapter.cpp` (registers the architecture, reads the pack, builds sessions),
the shared dense stack in `src/dense/`, `tools/convert/checkpoint_qwen35.py` (an NVFP4 checkpoint to a pack), and
`qwen35_checkpoint_round_trip` and `qwen35_linear_types_test`, which skip without a 27B pack.

### D1 (decided): Unsloth v3.1 GGUFs only

The dense path builds from Unsloth's Unsloth Dynamic GGUFs at the revision on disk (`4af51b35`): UD-IQ4_XS (14.35 GB)
and UD-Q4_K_XL (17.66 GB). No NVFP4 checkpoint, no QUASAR or NVIDIA requantization, no plain absmax requant that throws
away Unsloth's imatrix allocation. The NVFP4 converter stays in the tree as a tool and is not the path for any claim.

This closes BUILD_SPEC §11.1 in the direction the spec already chose: repack, do not requantize. The work it implies is
the qwen35 branch of the GGUF converter, which does not exist today.

### Measured state (2026-11-17)

Evidence: `evals/2026-11-17/halocline-dense-load` and `evals/2026-11-17/halocline-dense-quality`. Base PPL 6.7411.

| Arm | Weights | KV | Mean KLD | Top-1 | Prefill 36K |
|---|---|---|---|---|---|
| Halocline | UD-Q4_K_XL | int8 | 0.00582 | 96.67% | 3,234 t/s |
| llama.cpp `2cbe0ec` | UD-Q4_K_XL | f16 | 0.00573 ± 0.00023 | 96.54% | |
| Halocline | UD-IQ4_XS | int8 | 0.01618 | 94.31% | 3,383 t/s |
| llama.cpp `2cbe0ec` | UD-IQ4_XS | q8_0 / f16 | 0.01638 / 0.01614 | 94.35% / 94.34% | |

UD-Q4_K_XL is 2.8 times better than UD-IQ4_XS on mean KLD. Decode with MTP (`--spec 8`, sampled fixtures): IQ4_XS conv
113, prose 98 tok/s on native mmvq (96 / 77 on MMQ alone). mmvq costs +0.00015 mean KLD [+0.00005, +0.00025]
(fp16 activation scales); closing that gap is item Q2 below.

### D2: build and prove both packs (done 2026-11-17)

Write the qwen35 GGUF path, then convert both quants. Accept: every tensor's dequantized values match the GGUF bit for
bit (the converter's manifest hash per tensor), the load prints no warnings, and the pack fits with the chosen KV.

### D3: first load (done 2026-11-17, both quants; embedding kept in its GGUF type, 0.51 / 0.67 GiB)

Serve it with one prompt. Accept: all layers on the card, the token embedding in page-locked RAM (2.4 GiB), coherent
output, no CUDA error.

### D4: the BF16 gate

`halocline-bench --quality 100 --kld-base qwen38-27b-bf16-base.kld` (base PPL 6.7437, made from Unsloth's BF16 GGUF with
llama.cpp, sha256 `9ffb3bd9…`). Report mean, median, p99, top-1, PPL and the paired per-chunk interval for each quant and
each KV format that fits. Accept: the dense gate row exists, and the KV choice is made by whole-system KLD at depth,
then speed.

### D5: speed

Same fixtures and protocol as the sparse path, one repetition per arm, power limit recorded. Floors from the P0 run:
112 tok/s short decode, 118 at 36K, 93 at 137K; prompt 3,215 at 36K. Accept: the dense profile passes the floors with
the KV choice from D4.

### D6: identity and state (state round trip done 2026-11-17 on the real pack; build-to-build identity open)

Greedy identity across the last two builds, and `qwen35_checkpoint_round_trip` with the real pack. Accept: 900 of 900
tokens and a state round trip that reproduces bytes.

### D8: 262K context, MTP and vision on one card (open, the next engine job)

Measured with the engine's own planner, vision packs, `--spec 8` (`evals/2026-11-17/halocline-dense-quality`):
262,144 cells need 11,830 MiB at int8 KV (8,448 of it the KV, 1,024 the MTP drafter's fp16 KV, the rest state and
workspace). Free after the weights: 10,589 MiB on UD-IQ4_XS, 7,597 on UD-Q4_K_XL. **Neither quant fits 262K today.**
Q5 quants cannot fit it on 24 GB at any KV precision that keeps quality (UD-Q5_K_XL is 19.5 GiB).

The work, in order:

1. **K8V4 and 4-bit KV in the dense GQA kernels.** The dense kernels (`gqa_decode_attn`, `gqa_prompt_attn`, the
   cache write in `attn_qk_prepare`) take fp16 and int8 only; the sparse path's K8V4 and rotated Q4_0 live in its own QSA
   kernels (`src/models/qwen4exp/kernels/kv_q4.cu`). Port the rotation and quantizers, add the K8V4 and 4-bit variants,
   CPU-reference parity tests, the planner's `cell_bytes`, snapshots. A Hadamard rotation of V commutes with attention
   (the output is rotated back once per head), so V costs one transform per row. Research: QuaRot (arXiv 2404.00456)
   for rotation before 4-bit KV; KIVI (2402.02750) and KVQuant (2401.18079) for keys needing per-channel treatment and
   values tolerating per-token groups.

   *Status 2026-11-17 (W2, branch `d8-kv`): K8V4 is in the dense path (`--kv k8v4`, 1,632 B a cell), with the block and
   the quantizer moved to shared code (`include/halo/kernels/kv_q4_layout.hpp`, `src/kernels/cuda/kv_q4_layout.cu`) so
   the sparse and dense caches quantize through one function. The planner's own formula gives 164,094 cells with MTP and
   vision on UD-Q4_K_XL, and 184,694 once the drafter's KV is the 32K ring of item 2 below - the ~184K this item quotes.
   A pass over the format's bytes measured nothing yet: the KLD table, the long-context windows and decode and prefill
   are open. Rotated 4-bit K and V is not started (the task's second phase).*

   *Status 2026-11-17 (W7, branch `d8-q4`): rotated 4-bit K and V is in the dense path (`--kv q4_0`, `KvQ4`, 1,152 B a
   cell, conversation cache format 4). K is rotated after its rope at the cache write and quantized with the shared q4_0
   block, V as K8V4 has it; Q is rotated before the attention call in `win_attn`, `batch_attn` and `chunk_attn` by the
   same transform (`H . H = 1`, so Q K^T is the unrotated dot product), and each head's output is rotated back once. The
   write rotates K and V in block (one thread a dim). The prompt kernel is the K8V4 one with K's tile expanded from
   q4_0 blocks and one int32 accumulator a 32-dim block, 86,016 B of shared memory at 4 warps. The planner gives 220,035
   cells with the drafter at full context, 241,285 with `--mtp-kv int8`, 261,647 with the 32K ring (262,144 then short
   by 9 MiB, the trim item 3 names) and 265,176 with the ring and int8 (262,144 fits with 53 MiB to spare). No KLD, no
   decode or prefill number and no GPU parity run: the card was held by a production engine. `--mtp-kv q4_0` is refused,
   and both 4-bit formats refuse `--kv-resident`.*
2. **MTP drafter KV as a ring of its attention window** (32K) instead of full context: 1,024 MiB down to 128 at 262K.
   The drafter indexes cells by absolute position today.
   **2026-11-17:** both levers landed as opt-in flags, the default unchanged: `--mtp-kv fp16|int8` (the drafter's cache
   at 528 MiB at 262K in int8) and `--mtp-window N` (it reads the last N cells; its cache is a ring of N + 15, 128.05
   MiB at N = 32,768). A windowed session is not parked (a ring has no linear image). The acceptance A/B that would pick
   a default has not run.
3. **The quant is UD-Q4_K_XL** (maintainer decision, 2026-11-17: 2.8 times lower KLD than UD-IQ4_XS outweighs context).
   Its context by KV format, with ~4.6 GiB left for KV once the MTP KV is a ring: int8 ~142K, K8V4 ~184K, rotated 4-bit
   K and V ~261K (262K after a small workspace trim). **The deciding measurement:** UD-Q4_K_XL with K8V4 at 184K against
   rotated 4-bit K and V at 262K, on the long-context KLD gate (P0 item 3: windows at 64K, 137K, 184K and, for 4-bit, 261K)
   and on decode and prefill at 36K, 137K and each format's maximum. The maintainer's preference is K8V4 at ~184K if
   its decode is on par with the sparse path's served profile and NInfer; 4-bit K at 262K only if the measured KLD
   difference says it costs nothing that matters. **Whichever KV
   format wins at depth becomes the dense default, and this item stays open until that profile ships.** Until then the
   served profile is UD-Q4_K_XL, int8 KV, 122,880 tokens (the largest that fits with MTP and vision today).

   **2026-11-17 (W7):** the rotated 4-bit plan at 262,144 cells is 7,094 MiB with the drafter's KV as the 32K ring of
   item 2 (9 MiB short of the card's 7,597 MiB and the 512 MiB reserve) and 7,032 MiB with `--mtp-kv int8` too (53 MiB
   to spare): 262,144 fits once both drafter levers are on, without the workspace trim this item's "261K" line assumed.
   With the drafter at full context the same plan is 7,990 MiB (905 MiB over) or 7,494 (409 MiB over) at `--mtp-kv int8`.
   The numbers are the planner's own, `tests/kernels/dense_kv_planner_test.cpp`; nothing was measured on the card.

Accept: a 262K load with MTP and vision, the long-context KLD table, decode at 36K and 137K above the NInfer floors.

### Dense quality levers (each A/B'd on the 100-chunk gate with a paired interval; keep what wins)

- **Q1, FP32 residual stream.** The dense residual is BF16 (`src/dense/internal.hpp`), rounded about 128 times through
  64 layers; llama.cpp keeps it in FP32. Mixed Precision Training (arXiv 1710.03740) and Revisiting BFloat16 Training
  (2010.06192) show the loss of BF16 accumulation with nearest rounding. Parity with llama.cpp says the cost is small;
  the A/B says how small.
- **Q2, decode GEMV without fp16 activation scales,** or with no activation quantization (W4A16: decode is bandwidth
  bound, so dequantizing in registers costs ALU, not bytes). LLM.int8() (2208.07339) and SmoothQuant (2211.10438) show
  activation outliers are what int8 activations lose on at this scale.
- **Q3, exact Q8_0 GDN gates.** `ssm_alpha`/`ssm_beta` are rounded to BF16 at load (the one step that is not lossless).
  Run them as Q8_0 products and gate after.
- **Q4, upstream llama.cpp.** The vendored ggml/MMQ/MMVQ base is `3cf03257f` (2026-09-20), 1,431 commits behind master on
  2026-11-17. Port with the GPU suite and the gate after each: #31247 (Q8_1 correction sum), #29953 (MMQ out-of-bounds
  reads), #29683 (IQ4_NL short rows), #31533 (few-row MMQ for 2-24 rows: the 6-16 row verify windows), #30517 (faster
  MMVQ for low-bit types), #29948 (fused gate/up and GLU in MMQ), #29353 (chunked GDN). Open upstream, to watch: #31802
  (fused q8_0/q4_0/FP8 KV dequant in MMA flash attention), #31669 (GDN replay for MTP), #31634 (ordered GDN), #32454
  (NORM+MUL fusion), #32374 (64-bit MMQ offsets). Upstream #32102 merges qwen3.5 tensors on convert, so a future
  Unsloth GGUF may carry merged tensors: the converter must map or refuse them.

### D7: beat NInfer and llama.cpp on quality

The comparison is one model, one quant, one card, one reference:

| Arm | Weights | What is measured |
|---|---|---|
| Halocline dense | Unsloth UD-IQ4_XS or UD-Q4_K_XL, repacked losslessly | KLD gate, speed, identity |
| llama.cpp | the same Unsloth GGUF, `llama-perplexity` and `llama-server` with the same KV | KLD gate, speed |
| NInfer | its own 27B format (it requantizes; the bytes differ) | KLD if it exposes the same gate, speed, sampled outputs |

Rules for the comparison:

- The KLD base is the same file for every arm. Lower mean KLD and higher top-1 is the win.
- Weights differ between arms because NInfer requantizes. Report that difference instead of hiding it; the claim is
  about the served result, not about identical bytes.
- Speed runs alternate arms in one session (ABBA), power limit fixed, one engine on the card.
- Quality beyond KLD uses the task suite in P2 on the same prompts, because KLD near the baseline does not rank models.
- Accept: a table with all three arms and intervals, and a statement of which arm wins on KLD, on the task set, and on
  speed. If Halocline does not win, that is the result and the README says so.

## P0: close the cheap gaps in the published build

These run beside P1 because they need GPU time, not converter work.

1. **Strata KLD.** Score Strata 0.2.10 on the same frozen base with its own pack and KV. Without it the speed table
   compares two engines and the quality comparison is missing. Accept: the 100-chunk row for Strata.
2. **Sampling distribution.** Seeded runs, top-1 and top-10 frequency within the run-to-run interval of the reference
   build. Accept: the same prompts, the same seeds, the numbers recorded.
3. **Long-context KLD.** Score windows placed at 64K, 137K and 200K of context, per KV format. This is the gate that
   would catch int8 KV drifting at depth. Accept: the table.
4. **Tool-call breadth.** Twenty or more prompts with a whether-to-call label. Accept: accuracy against the reference
   path.

## P2: task-level eval (SWE-bench)

Distributional distance near the baseline does not predict task outcomes, so the repo needs a task gate. SWE-bench is
the right first one: long context, tool use and code editing in one loop.

- **Harness:** mini-swe-agent, one scaffold for every arm, OpenAI-compatible endpoint against `bin/halocline serve`.
- **Set:** SWE-bench Lite first (300 instances), then Verified (500). Pin the dataset revision and the images.
- **Settings:** temperature 0, one attempt per instance, 262,144 context, int8 KV, drafter on. Same flags for every arm.
- **Arms:** the sparse pack, the dense pack, then llama.cpp and NInfer on the same set if their harness support allows.
- **Report:** resolve rate with a 95% interval, tokens in and out, wall time per instance, and the KLD row for the
  same pack.
- **Blockers:** about 100 GB of image pulls, Docker with the NVIDIA runtime, hours per arm. Start with 25 instances.

Accept: a Lite run with a recorded resolve rate and no engine failures, repeated for each arm.

## P3: Qwen4 targets

Qwen4-Flash and Qwen4-27B are the stated targets. As of the research pass, Qwen4-Max is API-only and no Qwen4 weights
are public; Qwen4-27B's shape is a community lead. Do not plan against either until weights are on Hugging Face with a
config and a licence, the architecture matches a registered adapter or the diff is scoped, and a BF16 reference exists.
Then the order is P1 again: pack, load, KLD, speed, identity, task gate.

## P4: support and polish

- Remove the Windows and Docker code paths, or test them. Neither is supported today.
- Rewrite the carried-over user docs (`docs/configuration/`, `docs/models/`, `docs/precision/`); each still carries
  Basalt's banner.
- A release workflow: tag, archive, checksum, recorded toolchain.
- The formal trademark check for the name was never run; run it before commercial use.

## Order of work

D1-D4 and the D6 round trip are done (2026-11-17). Next: D8 (262K), then D5 against NInfer like for like, D7, and
the levers Q1-Q4. P0 runs in parallel because it needs the card, not a converter. D7 needs D3 to D5
first; P2 starts once the dense pack loads, so the task gate measures a model that runs. P3 waits on Qwen.
