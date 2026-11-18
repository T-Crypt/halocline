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
| Dense vs NInfer | prefill 2.4% under its floor at 36K, 6.5% over at 137K; decode 13-16% under (D5); 10x closer to BF16 on target-token dNLL at 65K and 140K (D7, 2026-11-18; NInfer has no logits for KLD) |
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

**Speed levers against NInfer (2026-11-17 lever map, ranked; each needs the like-for-like run first).** Before any
lever: run the P0 floor fixture (`lsw.py smoke --long`) through both engines on the same tokens, and add a per-stage
profile to the dense window (the sparse path has one, `src/models/qwen4exp/decode/verify/profile.cpp`). Our prose
(86.5 tok/s at 36K) already equals NInfer's own published prose at 64K (85.9), so the gap may sit mostly on structured
output; struct is 153.8 tok/s at 351 GB/s.

- **Tried and dropped 2026-11-18: KV-head-major prompt block order** (after llama.cpp #32407, which rotates K/V tile
  reads under stream-K; Halocline has no stream-K, so the analog was the prompt kernel's block order, branch
  `prompt-l2` 9770a55f, opt-in and bit-identical by test). ABBA on UD-Q4_K_XL, int8 and K8V4, 16K to 131K: it wins
  nowhere (-0.01% to -0.24%, run spread about 0.05%). Not merged (`evals/2026-11-17/halocline-prompt-l2`).
- **S1, token-tiled decode attention.** `decode_mma_kernel` holds 16 (row, head) pairs a block, so with 6 query heads a
  KV head it reads the window's KV `ceil(6T/16)` times (2x at T=4, 4x at T=9). NInfer's small-T kernel keeps all of a
  KV head's window rows in one CTA (`TokenTile * GroupSize <= 48`) and reads each cell once. The only lever whose value
  grows with depth (one redundant KV read is about 6 ms of a 30 ms window at 137K). Gate: bitwise against the current
  kernel at T = 1..9, then 36K / 137K / the format's maximum.

  *Built 2026-11-18 (W14, branch `s1-decode-attn`, no GPU run yet):* `decode_mma_window_kernel` puts every row of a
  window for one KV head in one block, R = ceil(M / 16) 16-row m16 groups a block (R <= 3, so T <= 8 at 6 query heads a
  KV head), and decodes each 32-cell K/V chunk into shared memory once for all of a block's groups, so a window reads
  its cells once per split instead of `ceil(6T/16)` times. Shared bytes a block: 46,080 at R = 1, 58,368 at R = 2,
  70,656 at R = 3 (two blocks an SM at R = 1, one from R = 2), registers 96 / 128 / 168 with no spills. A row runs the
  tiled kernel's statements inside the tiled kernel's 16-row group, so its output is bitwise that kernel's (the parity
  tests hold T = 1..9 against the tiled launch on fp16, int8, K8V4, the far cache, a ring and a batch window's
  segments). Windows past 48 rows (M > 48) keep the tiled launch. The GPU A/B at 36K and 137K is pending.
- **S2, one weight pass for the whole verify window at GEMV rate.** `kMmvqMaxRows = 5` (`src/dense/forward.cpp:59`):
  wider windows drop to MMQ, so struct (mean 7.5 rows) streams 351 GB/s against 758 GB/s for a T=1 window. Port
  llama.cpp #31533 (few-row MMQ, see Q4) first, then measure MMVQ with fp32 activation scales to 8 rows (also Q2).
- **S3, draft depth from the cost model.** Choose the window from `ms_per_window`, mean rows and acceptance (prose
  2.40 tokens at 27.7 ms, struct 7.43 at 48.3 ms); NInfer serves 3 drafts. One day, no kernel, no quality path.
- Later: rotated 4-bit KV (D8 item 3; a depth lever, never A/B it at short context), a block drafter (DFlash, arXiv
  2602.06036; needs a trained drafter), acceptance-targeted drafter training (arXiv 2602.23881). Measured as not worth
  it: the GDN decode recurrence (FP32 state on both engines, about 1.5% of a window) and a fused top-k sampler.

**Per-stage profile of the dense window (2026-11-18).** The dense path has the sparse path's instrument now: `--profile`
on the qwen35 path (a diagnostic flag, off by default) runs each decode window as launched (no graph replay) and stamps
its stages with CUDA events on the window's stream, with PDL off, so a stage's ms holds the launch gap before its own
first kernel. It sums a stage over the layers, aggregates per window width T (1..16) as windows, mean ms per window and
each stage's mean ms and share, and prints the table through `halo::log::info` (tag `halo dense profile`) every 200
windows and when the session ends; each line closes with the stages' sum against the window's wall and the gap between
them, so the launch gaps and host-side accept are visible. The stages: per attention layer its norm and q/k/v, `attn_qk_prepare` (the cache
write), `gqa_decode_attn` and the output projection; per GDN layer its projections, `dec_gdn_rec` and its output; the
FFN's gate/up and down; the LM head; sampling; and the MTP drafter's draft (`mtp_window`) and history (settle). Run it as
`bin/halocline serve PACK ... --profile`, or `halocline-bench ... -- --profile`; its numbers are not for A/B (uncaptured,
PDL off) and the batch window of several sessions is not stamped.

*Measured 2026-11-18 (`evals/2026-11-18/halocline-dense-profile`, served K8V4 flags, one server per workload):* under
`serve` the tables land in the server's log buffer (`GET /api/logs`, the stats page), not the terminal. attn-core is
1.8% of the window at short context, 10.7% at 36K and 27-29% at 137K, and it steps exactly with ceil(6T/16): 7.65 ms
for T = 1-2, 14.9 for T = 3-5, 22.3 for T = 6-8 at 137K. That is S1, measured: reading the KV once saves about 7% of
window time on prose and 11% on struct at 137K. The MTP drafter is 15-21% of every window at every depth; a draft step
costs 1.6-2.0 ms, of which the full 248,320-row head is 1.1 ms. NInfer's `--lm-head-draft` is a frequency-ranked
131,072-row draft head with an id map (FR-Spec, arXiv 2502.14856); that is the drafter lever (rows gathered as stored,
verify on the full head, so the output distribution is unchanged). GEMV at T = 6 steps up (MMQ past `kMmvqMaxRows`),
but mean T is 2.7-3.9, so S2 is third.

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

   *Measured 2026-11-17 evening (`evals/2026-11-17/halocline-d8-kv`):* K8V4 on UD-Q4_K_XL at 36K scores 0.00676 mean
   KLD against int8's 0.00582, paired +0.00093 [+0.00074, +0.00113], top-1 -0.28 points; decode within 2%; prefill
   -2.1% / -4.6% at 16K / 36K after the 8-warp prompt shapes (`7e0f43c4`; -6.7% / -12.7% at 4 warps). The K8V4 prompt
   kernel had a sign bug for blocks with a positive largest value (`2b6095a4`). The planner reaches 175,629 cells with
   `--mtp-kv int8`. Rotated 4-bit K and V (`--kv q4_0`, 1,152 B a cell) is on branch `d8-q4`: GPU parity passes
   (decode 0.00082, prompt 0.0234 against fp64), planner 241,285 cells with `--mtp-kv int8` and 265,176 adding the 32K
   ring (262,144 fits with both, 53 MiB to spare). **Its KLD at 36K** (`evals/2026-11-17/halocline-d8-kv`, arm `q4`):
   0.00779, paired +0.00197 [+0.00167, +0.00228] over int8 and +0.00104 [+0.00082, +0.00125] over K8V4, top-1 96.07%;
   decode on par; prefill 3,151 / 2,745 at 16K / 36K (-8% / -15%, 4 warps). Not merged into `d8-integration` yet: it
   changes shared kernels non-bitwise (W7's SASS diff), so it carries the sparse gate before it merges.
2. **MTP drafter KV as a ring of its attention window** (32K) instead of full context: 1,024 MiB down to 128 at 262K.
   The drafter indexes cells by absolute position today.
   **2026-11-17:** both levers landed as opt-in flags, the default unchanged: `--mtp-kv fp16|int8` (the drafter's cache
   at 528 MiB at 262K in int8) and `--mtp-window N` (it reads the last N cells; its cache is a ring of N + 15, 128.05
   MiB at N = 32,768). A windowed session is not parked (a ring has no linear image). The acceptance A/B that would pick
   a default has not run.
   **Measured 2026-11-17:** `--mtp-kv int8` leaves the 100-chunk gate's scored rows byte-identical to fp16 (the verify
   pass decides every token) and decode within 2%, so it is the recommended setting with every format; the ring stays
   a last resort because it gives up parking and the conversation cache.
   **2026-11-18:** the ring was broken past one lap: `gqa_decode_attn` swept it from chunk 0, so beyond position
   `cap` the drafter read no cell (NaN logits, "the drafter failed" at depth 140,288). Fixed on branch `mtp-ring-fix`
   (sweep from the window's first chunk; a ring case in `gqa_attn_parity` reads max abs 0 against an absolute cache).
   Only the dense drafter passes a ring, so the main decode and the sparse path are unchanged
   (`evals/2026-11-17/halocline-d8-depth`).
3. **The quant is UD-Q4_K_XL** (maintainer decision, 2026-11-17: 2.8 times lower KLD than UD-IQ4_XS outweighs context).
   Its context by KV format, with ~4.6 GiB left for KV once the MTP KV is a ring: int8 ~142K, K8V4 ~184K, rotated 4-bit
   K and V ~261K (262K after a small workspace trim). **The deciding measurement:** UD-Q4_K_XL with K8V4 at 184K against
   rotated 4-bit K and V at 262K, on the long-context KLD gate (P0 item 3: windows at 64K, 137K, 184K and, for 4-bit, 261K)
   and on decode and prefill at 36K, 137K and each format's maximum. The maintainer's preference is K8V4 at ~184K if
   its decode is on par with the sparse path's served profile and NInfer; 4-bit K at 262K only if the measured KLD
   difference says it costs nothing that matters. **Whichever KV
   format wins at depth becomes the dense default, and this item stays open until that profile ships.** Until then the
   served profile is UD-Q4_K_XL, int8 KV, 122,880 tokens (the largest that fits with MTP and vision today).
   **Measured 2026-11-18** (`evals/2026-11-17/halocline-d8-depth`, 16 chunks x 1,024 rows after the prefix): K8V4
   over int8 +0.00072 [+0.00050, +0.00095] at 65,536 and +0.00066 [+0.00050, +0.00082] at 140,288, the same cost as
   at 36K; K8V4 alone at 174,080: 0.00654 mean KLD, 96.62% top-1. Rotated 4-bit K and V at 65,536: +0.00185
   [+0.00135, +0.00234] over int8, 2.6 times K8V4's increment. Its 261,119 arm did not run: the protocol needs
   262,152 cells there (the engine keeps 8 cells past the last generated token). Decode and prefill at each format's
   maximum are still open.
   **Decided 2026-11-18 (maintainer): K8V4 is the dense default.** Served profile UD-Q4_K_XL, `--kv k8v4 --mtp-kv int8`,
   175,616 tokens, MTP and vision; its llama-swap smoke passes, the 137K needle included. The engine's own default for an
   unset `--kv` is still fp16. 262K stays open (rotated 4-bit, or a smaller drafter cache).
   **Open, deferred (maintainer 2026-11-18, "we will fix it later"):** make K8V4 the engine's default for an unset
   `--kv` on the dense path (`kv_format` in `src/models/qwen35/adapter.cpp`, and the "fp16 (the default)" message).
   The trap: K8V4 has no streamed form, so a run that passes `--kv-resident` without `--kv` would be refused; resolve
   that (streaming falls back to int8, or the refusal names `--kv int8`) and run the GPU suite before the change lands.

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

**NInfer quality, measured 2026-11-18** (`evals/2026-11-18/ninfer-depth-ppl`). NInfer exposes no logits, so this is
target-token dNLL (mean NLL of the observed token minus BF16's, paired per chunk), not KLD, on the depth bases' own
rows (16 x 1,024 at 65,536 and 140,288, the same tokens round-tripped). Served against served: Halocline K8V4 +0.0018
and +0.0020; NInfer `rk4v4-e8` +0.0199 [+0.0157, +0.0241] and +0.0201 [+0.0145, +0.0257], paired +0.018 against
Halocline at both depths. NInfer with int8 KV is still +0.0172 at 65K: most of its loss is its weight conversion, not
its 4-bit cache. Halocline wins this axis by about 10 times; speed is the open one (D5).

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
