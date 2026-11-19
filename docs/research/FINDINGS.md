# Findings

The ten notes in `sm89-sparse-dense-engine-quality/` answer one question each. This page is the short version: the
decision, what the research found, what the engine does, and the evidence. Where a claim is a measurement, it names the
eval. Where it is a paper's result, it links the paper. Where the notes could not establish something, it says so.

## The decision table

| Decision | What the research found | What the engine does | Evidence |
|---|---|---|---|
| Start from the `qwen4_exp` architecture | Qwen4-Max is API-only with no weights; the verified open artifact is Qwen3.8-Flash-Next | the adapter targets `qwen4_exp`; Qwen3.8-27B stands in for a dense model | [config at `de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json), [tech report](https://arxiv.org/abs/2608.30320) |
| Keep Unsloth's quantized bytes, mixed encodings included | a 4-bit MoE cannot fit 24 GB at BF16; re-quantizing adds rounding the file does not have | the pack copies quantized tensors byte for byte; 195 `hc_*` tensors are cast to BF16 as an interim step | `SPECS.md`, [`evals/2026-11-17/halocline-kv-kld`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-kv-kld) |
| int8 KV | the literature gradient is int8, then E8-compressed formats; published costs put int8 and fp16 within noise | `--kv int8` | [KLD table](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-kv-kld), [arXiv 2610.15384](https://arxiv.org/abs/2610.15384) |
| MTP drafter at window 8 | a drafted method is lossless only with a correct verifier; acceptance is workload-dependent | `--spec 8`; window 12 accepts no further than 8 | [draft window table](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-kv-kld) |
| Experts in VRAM and page-locked RAM, none on SSD | placement is not math if the same bytes reach the same kernels | 18.2 GiB in VRAM, 38.0 to 41.3 GiB pinned in RAM, 0 on SSD | [`halocline-first-load`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-first-load), [`halocline-greedy-identity`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-greedy-identity) |
| sm_89 kernels written for Ada | Ada has int8, BF16 and FP16 `mma.sync`, `cp.async` and 99 KB opt-in shared memory; it has no FP4 tensor cores, no clusters, no TMA and no PDL | the engine compiles sm_89 only, with Ada replacements for the cluster, TMA, FP4-convert and FP8 paths | `ADA_PORT_NOTES.md`, [arXiv 2608.10103](https://arxiv.org/abs/2608.10103) |
| KLD plus canaries, not KLD alone | KLD ranks a quant cohort against a baseline but loses that link near the baseline | every serve also runs needle retrieval at 36K and 137K and a tool-call check | [arXiv 2606.19558](https://arxiv.org/abs/2606.19558), [arXiv 2608.06564](https://arxiv.org/abs/2608.06564) |

## Weight format

The pack keeps the published quantized tensors. The reason is arithmetic: a 4-bit weight file already contains its
rounding, and re-quantizing it adds a second, uncounted one. The file in use is Unsloth's UD-IQ4_XS v3.1, four shards,
93,672,359,738 bytes. It is a mixed-encoding file, not a uniform IQ4_XS file; the note's parse puts IQ4_XS at 17.1% of
the bytes, a figure the note itself marks as a single-pass parse. What matters for the engine is that the bytes are
carried unchanged and that the gate scores the result.

The 4-bit build's KLD against BF16 is 0.0820 with 90.45% top-1 agreement at 100 chunks. The KV sweep's spread is 0.0065,
so the gate can separate KV formats but does not isolate the weight quant's own share.

The `hc_*` cast is the one place the engine changes bytes. 195 tensors are Q8_0 in the GGUF and are cast to BF16 at
conversion (`--compat-bf16`), because the engine needs its own `hc8` copies before the cast. The KLD numbers include the
cast. It is an interim step, and reopening it is open work.

## KV cache and context

Four formats were scored on the same pack and text, 100 chunks, paired per chunk against int8:

| KV | mean KLD | top-1 | paired dKLD [95% CI] | prefill 131K |
|---|---|---|---|---|
| int8 | 0.08195 | 90.45% | reference | 6,019 t/s |
| fp16 | 0.08230 | 90.33% | +0.00035 [-0.00061, +0.00131] | 2,997 t/s |
| k8v4 | 0.08621 | 90.26% | +0.00426 [+0.00307, +0.00545] | 5,694 t/s |
| q4_0 | 0.08843 | 90.03% | +0.00647 [+0.00499, +0.00795] | 5,700 t/s |

fp16's interval holds zero, so this gate cannot place it closer to BF16 than int8, and fp16 takes VRAM from the expert
cache, which halves the 131K prefill. `--kv int8` is the setting.

The notes record the same order from other engines and from the paper on E8 lattice key quantization: lower-bit KV is
decision-sensitive in a way weight-only changes are not. That is why the gate is paired per chunk and why long-context
needle checks run beside it.

## Speculative decoding

MTP, DFlash and DSpark all verify against the target model, so they are lossless when the verifier is correct. What
varies is acceptance, which depends on the prompt, the sampling and the context: the notes record MTP acceptance from
about 65% on prose to about 97% on structured output in this engine's own fixtures. The published draft-window sweep
picked 8: window 5 is 11% slower on structured output, and window 12 accepts no further than 8 while costing another
draft step.

The drafter adds 0.77 GiB to the pack (`mtp/dense` 0.108 GiB and `mtp/experts` 0.659 GiB) and takes that VRAM from the expert cache. At 137K the prompt rate is 13% lower with the
drafter on than without it, and decode is 32% higher (121.0 against 91.3 tok/s), because decode is where the drafting
pays.

## Expert placement and memory

The notes' conclusion is that offload is capacity and latency engineering, not a numeric-precision change, provided the
same bytes reach the same kernels. The engine's gate for that is greedy identity with the placement pinned, and the
identity run compares the build from before the removals with the published build.

The binding constraint on this machine is RAM, not VRAM: 41 GiB of experts are page-locked, and about 4 GiB of system
RAM stays free with the model loaded. Under host memory pressure, the notes record prompt rates falling 30 to 40% and
one run killed by the out-of-memory daemon. That is why every speed number in this repo comes from a quiet host and
states its power limit, and why the reference machine's memory plan is in `SPECS.md`.

## Kernels on Ada

Ada's tensor cores take int8, BF16 and FP16 `mma.sync`, not the block-scaled FP4 and FP8 forms Blackwell added. The port
notes record what that means per kernel:

- The dense FP4 product decodes E2M1 codes to FP16 and runs FP16 `mma.sync` with fp32 accumulation. The products are
  exact; the reduction order differs from a Blackwell card's, so a 4090's prompt rows are not a 5090's bits.
- FP8 W8A8 is not exact on Ada. At K 6144 the E4M3 `mma.sync` form loses 1.2e-3 of the sum of |w a| against a 1e-5
  test threshold, so the engine decodes the codes to FP16 and runs FP16 `mma.sync` with fp32 accumulation. That costs
  throughput: 134 to 161 TFLOP/s against 220 to 268 on the FP8 tensor cores.
- `cp.async` pipelines cover the copies. Thread-block clusters, TMA, bulk async copies and programmatic dependent
  launch do not exist on sm_89, and their kernels are compiled out and never selected.

The kernel-level evidence is in `ADA_PORT_NOTES.md` and the SASS hashes in `gates/`; each kernel has a parity test in
`tests/` against a CPU or double-precision reference.

## Dense and sparse

The verified open model is sparse: 512 experts, 10 routed plus 1 shared. The dense path runs too: Qwen3.8-27B stands in
until Qwen4-27B weights exist, and those weights are a community lead, not a published artifact. The switch rule the notes settle on is capacity: a sparse model wins when inactive experts can live
in RAM and latency is acceptable, and a dense model wins when every parameter must be read per token and the model fits.

## Validation method

The notes' method section concludes that no single metric proves a BF16 match. The engine uses four layers:

1. SASS hashes per object and per function, so a change that should not touch device code provably does not.
2. Greedy identity against the previous build with placement pinned, for changes that must not alter output.
3. The KLD gate against a frozen BF16 base, reported as mean, median, p99, top-1 and a paired per-chunk interval.
4. Canaries the gate cannot see: needle retrieval at 36K and 137K, and a tool-call check.

The two papers behind the fourth layer matter. KLD tracks benchmark scores across a quant cohort and loses that link near
the baseline, and low-bit quantization flips whether to call a tool more often than which tool. A third paper separates
determinism from invariance: two runs of one build matching row for row does not mean another build matches.

## What the research did not establish

- No public single-4090 run of this model existed when the notes were written. The closest public evidence was a
  four-card sm_86 llama.cpp run.
- No sm_89-specific numerical study of the model was found. The kernel evidence is this repo's own parity tests and SASS
  hashes.
- The IQ4_XS census in the note is a single parse. The byte size is verified; the per-encoding shares are not.
- Qwen4-27B's shape comes from keynote slides and kernel strides, not a config. The notes label it T4 and keep it out of
  the spec.
- No KLD comparison of Strata on this host exists. The speed table against Strata is speed only. Strata was retired on
  2026-11-18, so none is planned; its rows stay as history.
