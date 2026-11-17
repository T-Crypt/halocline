# IQ4_XS, BF16 to 4-bit quality preservation, and IQ4_XS as the engine's first-class quant

**Date:** 2026-11-16
**Scope:** one NVIDIA Ada GPU, sm_89, RTX 4090 24 GB. Nothing here assumes sm_86, sm_90, sm_100, sm_120 or a multi-arch build.
**Role:** evidence for `BUILD_SPEC.md` §3 (kernel replacement map), §4 (quality contract) and §7 item 3 (reference product quant). Where this note and `BUILD_SPEC.md` differ, `BUILD_SPEC.md` wins.

Evidence labels follow the research base convention. T0 is our own measurement or our own read-only census. T1 is reproduced, T2 is a paper, T3 is an official artifact, teardown or vendor document. T4 is a community lead and is not evidence. `draft` means written by a model and not reviewed by the maintainer.

All local numbers below come from read-only work: parsing the four GGUF shards of the live artifact, reading the fork at `~/wt/basalt-ada-lift`, and reading `state/changes.md`. No GPU job ran, nothing was downloaded, no service was touched. Round numbers are marked as derived.

**Verification note (2026-11-16, follow-up pass).** The on-disk size was rechecked with `stat` on all four shards: 93,672,359,738 bytes, which is 87.24 GiB or 93.67 GB, not the 84.5 GiB / 90.7 GB stated below. The per-encoding census table and its percentages are the lane's own parse and were not independently reproduced; a separate reparse in the follow-up pass hit a metadata layout that the local parser could not read. Treat the encoding shares as lane-reported T0, not as a confirmed byte census. The conclusions in sections 3 to 5 rest on the mixed-encoding structure and the published artifact table, both of which are also supported by T3 sources, so the size correction does not change them.

## 1. What this note settles, and how to read it

The spec asks for a reference product quant (§7 item 3) and a fallback kernel for 4-bit weights (§3 item 4). Both assume IQ4_XS is a uniform format chosen for a model. Section 2 shows what BF16 to IQ4_XS costs on this family, using the only published BF16-referenced table for it, and then shows that our live artifact is a mixed-encoding file where IQ4_XS is 17.1% of the bytes. Section 3 compares ingest, re-quantize and own-format, and argues on the artifact we actually run. Section 4 lists the techniques with a verdict each. Section 5 lists the places the spec now says something the artifact contradicts. Section 6 lists what a GPU run would settle and what it would not.

Two numbers frame everything after: 84.5 GiB on disk for 4.102 bits per weight, and 90 to 100 t/s measured decode on this host. The load path is not where quality is won or lost; the choice of bytes per tensor is.

## 2. BF16 -> IQ4_XS quality preservation

### 2.1 What the format actually stores

An IQ4_XS block is 256 weights in 136 bytes, which is 4.25 bits per weight, plus whatever the loader rounds up at row ends. The layout is fixed in llama.cpp's shared header (`ggml/src/ggml-common.h`, read 2026-11-16, T3):

```c
typedef struct {
    ggml_half d;              // one fp16 scale for the 256-element super-block
    uint16_t  scales_h;       // high bits of eight 6-bit sub-block scales
    uint8_t   scales_l[QK_K/64];  // four bytes, low bits of the same eight scales
    uint8_t   qs[QK_K/2];     // 128 bytes, two 4-bit indices per byte
} block_iq4_xs;               // static_assert: 136 bytes
```

Each 4-bit index addresses a fixed 16-entry table of signed integers, `kvalues_iq4nl = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}` (same file, T3). The values are non-uniform: denser near zero, wider at the tails, which is the point of a non-linear codebook.

Three properties follow, and they matter more than the bit count:

1. **The scale grid is per 32 weights, not per 256.** Eight 6-bit scales per super-block, each signed by an offset, multiplied by the fp16 super-scale. A linear Q4_0 block spends 4.5 bits per weight on one fp16 scale per 32 and no sub-scaling; IQ4_XS spends 4.25 and gets a finer grid plus a codebook.
2. **The index is not a number.** You cannot feed a nibble into an integer matrix unit and get the right value. A table lookup has to happen first. This is the single fact that shapes every kernel decision in section 3.
3. **Nothing about the format is model-specific.** The block is the same for every tensor, so quality differences between two IQ4_XS files come from which tensors were quantized, and with what calibration data, not from the block.

### 2.2 The one controlled measurement for this model family

Unsloth publishes a BF16-referenced quant table for Qwen3.8-Flash-Next, the sparse model this project targets (`https://unsloth.ai/docs/models/qwen3.8-next`, page read 2026-11-16, T3 official artifact documentation). Their method is stated on the same page: 300 held-out examples, greedy argmax over 32 tokens, BF16 against each quant.

| quant | GB | top-1 | mean KLD | 99.9% KLD |
|---|---|---|---|---|
| UD-IQ1_S | 72.5 | 77.325 | 0.396070 | 7.2126 |
| UD-Q2_K_XL | 78.9 | 82.715 | 0.224607 | 4.9121 |
| UD-IQ3_XXS | 82.0 | 85.414 | 0.165120 | 4.0375 |
| UD-Q3_K_XL | 90.0 | 88.315 | 0.106504 | 3.0538 |
| **UD-IQ4_XS** | **93.7** | **89.554** | **0.083630** | **2.3677** |
| UD-Q4_K_XL | 111.3 | 92.255 | 0.046893 | 1.5468 |
| UD-Q5_K_XL | 158.3 | 93.680 | 0.030415 | 1.0036 |
| UD-Q6_K_XL | 169.2 | 94.089 | 0.027091 | 0.8416 |
| Q8_0 | 188.2 | 94.122 | 0.026574 | 0.8118 |

Read it as: the IQ4_XS file keeps 89.6 of Q8_0's 94.1 top-1 points while using 93.7 GB against 188.2 GB of disk, and its mean KLD is 3.1x the Q8_0 file's. UD-Q4_K_XL costs 17.6 GB more for 0.037 lower mean KLD (0.0469 against 0.0836) and 2.7 more top-1 points.

Caveats, all of which matter for a quality gate:

- The row is a whole file, not a tensor. Most of its bytes are not IQ4_XS (section 2.3), so this table cannot be used to argue that the IQ4_XS block format is or is not good.
- Our local copy is 84.5 GiB, which is 90.7 GB, not 93.7 GB. The local four shards hold 1224 tensors and zero MTP tensors; Unsloth ships an MTP head as a separate optional file. The card's row is therefore not byte-identical to the artifact we run, and which revision it measured is not stated on the page.
- Unsloth's own calibration and evaluation are on agentic, coding and multilingual traffic. A wikitext-2 KLD gate, which is what `BUILD_SPEC.md` §4.2 specifies, measures a different distribution.

### 2.3 Our census of the live artifact: the file is not "an IQ4_XS model"

Read-only parse of `/mnt/models/Strata/models/unsloth-UD-IQ4_XS-v3.1/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001..00004-of-00004.gguf` (T0, this session):

| encoding | tensors | GiB | share of bytes |
|---|---|---|---|
| IQ4_NL | 14 | 32.54 | 38.5% |
| Q5_0 | 33 | 16.11 | 19.1% |
| IQ3_XXS | 56 | 15.38 | 18.2% |
| IQ4_XS | 35 | 14.43 | 17.1% |
| Q6_K | 213 | 2.18 | 2.6% |
| IQ2_S | 6 | 1.76 | 2.1% |
| Q8_0 | 284 | 1.34 | 1.6% |
| Q2_0 | 2 | 0.44 | 0.5% |
| F32 | 557 | 0.29 | 0.3% |
| BF16 | 24 | 0.04 | 0.0% |
| **total** | **1224** | **84.5** | **4.102 bpw** |

Grouped by what the tensor does (T0, same parse):

| role | GiB | share | elements | effective bpw | encodings in use |
|---|---|---|---|---|---|
| MoE gate/up experts | 31.25 | 37.0% | 80.5e9 | 3.33 | IQ3_XXS 56, IQ4_XS 34, IQ2_S 6 |
| PLE / n-gram table | 26.82 | 31.7% | 51.2e9 | 4.50 | IQ4_NL 1 |
| MoE down experts | 22.27 | 26.3% | 40.3e9 | 4.75 | Q5_0 33, IQ4_NL 13, Q2_0 2 |
| attention | 1.94 | 2.3% | 2.4e9 | 6.87 | F32, Q6_K, Q8_0 |
| routing and misc | 1.22 | 1.4% | 1.0e9 | 10.41 | F32, Q8_0, BF16 |
| token embd / output | 0.81 | 1.0% | 1.3e9 | 5.42 | Q8_0, Q6_K, IQ4_XS |
| shared expert | 0.20 | 0.2% | 0.2e9 | 7.25 | Q6_K, Q8_0, F32 |

The file's own metadata says `general.file_type = 30`, which llama.cpp defines as `LLAMA_FTYPE_MOSTLY_IQ4_XS` with the comment "except 1d tensors" (`include/llama.h`, read 2026-11-16, T3). "Mostly IQ4_XS" is a label for the 2-D tensors llama.cpp chose, and here those tensors are 17.1% of the bytes. The same header carries `quantize.imatrix.dataset = unsloth_calibration_Qwen3.8-Flash-Next.txt`, `quantize.imatrix.chunks_count = 45`, `quantize.imatrix.entries_count = 926`, `quantized_by = Unsloth`, so this is an imatrix-calibrated file, not a blind quantize.

Three things this changes:

- **The quality was bought with allocation, not with the block format.** Gate/up experts run at 3.33 bpw, below IQ4_XS. Down experts run at 4.75, above it. Attention runs at 6.87, effectively lossless. Uniform IQ4_XS everywhere would spend more bytes on gate/up and fewer on down, which is the opposite of what the published numbers reward.
- **The largest single tensor is a lookup table at IQ4_NL.** `per_layer_token_embd.weight` is `[160, 320001536]`, 51.2e9 elements, 26.82 GiB, 31.7% of the file. Unsloth publishes a relative RMSE against BF16 for that table specifically: IQ4_NL 0.0757, Q5_1 0.0376, Q8_0 0.0053, BF16 0 (T3, same model page). Their stated reason for stopping at 4-bit is that the table has a random access pattern and quantizing it harder damages the model; the smaller quants in the table above still keep it at 4-bit or better.
- **Attention precision is not the question anyone should be arguing about.** It is 2.3% of the bytes at 6.87 bpw. Moving it to IQ4_XS would save about 0.7 GiB and risk the one part of the model where a small KLD change is most visible.

### 2.4 Verdict on preservation

BF16 to this artifact costs about 4.6 top-1 points and 0.0836 mean KLD against BF16, per Unsloth's own measurement of the family, with the tail at 2.37. Inside that number, IQ4_XS blocks account for roughly a sixth of the bytes. The quality argument for IQ4_XS as a first-class format is therefore not "IQ4_XS is nearly BF16". It is "this artifact's allocation, which includes IQ4_XS blocks, is the best published quality per byte for this model, and an engine that requantizes it throws the allocation away".

## 3. IQ4_XS as the engine's first-class quant

The build spec leaves the reference quant open (§7 item 3) and asks the kernel map to fall back to "IQ4_XS ggml MMQ" where NVFP4 is not available (§3 item 4). Both are written as if IQ4_XS were a uniform format chosen for a model. It is not. This section compares the three ways to make IQ4_XS first-class and argues from the artifact we actually run.

### 3.1 Option A: ingest GGUF directly, keep the bytes

The fork already does this (T0, code read this session):

- `include/basalt/artifact/gguf_reader.hpp` names the IQ types and reads them as blocks, with `dequant.hpp` carrying scalar reference dequantizers including IQ4_XS.
- `src/kernels/cuda/iq_kernels.cu` and `native_mmvq.cu` carry the llama.cpp vec_dot and dequant paths for the IQ family.
- `src/models/qwen4exp/prefill/moe_fused_iq.cu` runs a fused int8-MMA expert path over IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS for gate/up, and Q2_0 and IQ4_NL for down. That is the exact set this artifact uses, minus Q5_0 down experts.
- `docs/model-file.md` describes the `.basalt` v2 pack, which copies quantized tensors bit for bit and can be built from a GGUF, so the container adds a header and a layout, not a requantization.

Quality risk of A: none at load. Every weight is the byte Unsloth shipped, including the 3.33 bpw gate/up allocation and the IQ4_NL PLE table. Whatever error the artifact has, the engine inherits unchanged, and that error is the published one.

Cost of A: a kernel table keyed by encoding instead of one code path, which the fork already carries. The missing piece is Q5_0 down experts at 16.11 GiB, 19.1% of the file.

### 3.2 Option B: re-quantize into a groupwise format like NInfer's `.ninfer`

NInfer's own format stores Q4/Q5 projections plus Q6/Q8 vocabulary with a group of 64 weights and one fp16 scale per group, produced by `grouped_absmax` (`~/ninfer-4090/docs/weight-conversion.md`, T3 draft, not reviewed). Its published 4090 A/B compares that artifact against llama.cpp UD-Q4_K_XL at similar size (`~/ninfer-4090/docs/llamacpp-comparison.md`, T3 draft).

Quality risk of B: the evidence does not transfer. Uneven-group scaling with a linear grid at 5 bits is a different transform from a 16-entry codebook with 6-bit sub-block scales at 4.25 bits, and requantizing loses the imatrix-driven per-tensor allocation. There is no published BF16-referenced KLD for a groupwise 4090 artifact on this model, so adopting B means re-deriving the quality case from scratch.

Cost of B: a converter, a new kernel family, and a fresh KLD qualification against the same base. That is the largest single piece of work in this note, and it buys a bytes-per-weight number we can already get by choosing a different Unsloth file.

### 3.3 Option C: define an own packed layout for IQ4_XS

Only worth it if it buys speed that GGUF blocks cannot. On sm_89 it does not, for two reasons.

**Decode is not an MMA problem.** At batch 1 the expert path is a matrix-vector product whose cost is reading bytes. Using the per-role bpw above and top-10 of 512 experts over 48 layers, the activated expert traffic is about 1.05 GiB per token (derived, not measured). The measured product runs 90 to 100 t/s on this host (T0, `state/changes.md`), so the engine moves roughly 95 to 105 GB/s of expert bytes per second. The 4090's memory bandwidth is about 1 TB/s by vendor spec, and the machine has 64 GB of system RAM. The number that fits is host DRAM bandwidth, not VRAM bandwidth, which is consistent with the residency split the live Strata args use (`--resident-budget-gib 40`, `--pcie-frac 0.20`). If that reading is right, decode throughput is set by activated bytes per token divided by host memory bandwidth, and no repacking of the 4-bit indices changes either term.

**Prefill does use tensor units, but the codebook blocks the 4-bit path.** The fork feeds IQ4_XS into `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` after expanding nibbles through `kvalues_iq4nl` into int8 (`src/models/qwen4exp/kernels/dense_tc.cu`, type 23, T0). PTX does have a 4-bit MMA: `.u4/.s4` with `.m16n8k32` and `.m16n8k64` "requires sm_80 or higher", so it exists on Ada (PTX ISA, read 2026-11-16, T3). It is also useless here, because the operand would have to be the codebook value, not the index, and expanding the index is the work. FP8 `e4m3` MMA also exists on sm_89 ("requires sm_89 or higher", same document, T3), and it has the same problem: the weight is not an e4m3 number.

The one place a different integer width could pay is the activation side, not the weight side, and that is a measurement, not an assumption (section 6).

### 3.4 Recommendation

Make IQ4_XS first-class by ingesting GGUF blocks unchanged, and describe the format as what it is: one of ten encodings a product artifact can contain, with a per-encoding kernel row. The engine's quality claim then reduces to "we decode the published artifact exactly", which is checkable in one test (bit-exact dequant of a sampled tensor against the GGUF), instead of "we reproduced Unsloth's allocation", which is not checkable at all.

Concretely: keep the current GGUF and `.basalt` path, add the missing Q5_0 down-expert row, and treat a groupwise or own-format path as a later experiment gated on a measured prefill win (section 6).

## 4. Technique ledger

Verdicts are ADOPT, ADAPT, REJECT, or MEASURE FIRST. "Source" carries the tier and status of the claim the row rests on.

| technique | source (tier/status) | sm_89 benefit | quality risk BF16 to 4-bit | port cost | verdict |
|---|---|---|---|---|---|
| Direct GGUF ingest of the mixed-encoding UD artifact, copied bit for bit into `.basalt` | T0 code (fork `gguf_reader.hpp`, `docs/model-file.md`) plus T0 census of the live 4-shard file | none at decode, removes any requant step | none: the published artifact error is inherited unchanged | low, already present | **ADOPT** |
| Per-encoding kernel table (IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S, Q2_0, Q5_0, Q6_K, Q8_0) instead of one quant path | T0 census (10 encodings, 1224 tensors) | matches the artifact; avoids a slow generic fallback on 19% of bytes | none | medium: Q5_0 down experts are the gap | **ADOPT** |
| Fused int8-MMA expert prefill over IQ gate/up plus Q2_0/IQ4_NL down | T0 code (`moe_fused_iq.cu`, covers exactly this artifact minus Q5_0) | prefill throughput on the 37% of bytes that are gate/up | small: the path quantizes activations to int8 per 32 features, which is the real error term | low, present | **ADOPT** |
| IQ4_XS through `mma.sync m16n8k32 s8.s8.s32` with codebook expansion, scales applied in fp32 | T0 code (`dense_tc.cu` type 23), T3 PTX ISA for the shape | prefill only; decode is memory bound | none beyond the int8 activation quantization already counted | low, present | **ADOPT** |
| 4-bit (`.u4/.s4`) tensor-core MMA for linear 4-bit weights | T3 PTX ISA: `.m16n8k32`/`.m16n8k64` `.u4/.s4` requires sm_80+ | unverified on Ada consumer parts; the 2x over INT8 claim has no source here | not applicable to a codebook format; only touches linear Q4_0/Q4_K | high: new kernel family | **REJECT** for IQ formats, **MEASURE FIRST** only if a linear 4-bit reference quant is ever adopted |
| FP8 `e4m3` MMA for weights | T3 PTX ISA: `e4m3/e5m2` MMA requires sm_89+ | exists on Ada | would require requantizing, which reintroduces the allocation problem | high | **REJECT** for weights; KV use is covered by the KV note |
| Keep the PLE / n-gram table at 4 bits (IQ4_NL) rather than dropping it | T3 Unsloth docs (4-bit minimum, random access, relative RMSE 0.0757), T0 census (31.7% of bytes, 51.2e9 elements) | none: 1440 bytes per token of traffic, so bits do not buy speed | large: this is the biggest single quality lever in the file | low | **ADOPT** |
| Unsloth Dynamic per-tensor allocation with imatrix calibration | T3 Unsloth docs (`dynamic-3.0-ggufs`, v3.1, 45 chunks, 926 entries), T3 research-base lane `quantization` (living) | decides bytes per tensor, which is what sets decode speed | it is the thing that keeps 4-bit quality; treating it as noise loses 4.6 top-1 points | none: it is a property of the artifact | **ADOPT** as artifact choice, not as code |
| Re-quantize to a NInfer-style groupwise format (group 64, fp16 scale, `grouped_absmax`) | T3 draft (`doc:prep4qwen/ninfer-4090-udpsendtofailed`, `~/ninfer-4090/docs/weight-conversion.md`) | uniform kernels, simpler tiles | not measured for this model; discards the UD allocation | high: converter, kernels, full KLD requalification | **REJECT** for v1 |
| GSQ/RCO style per-tensor type search under a size budget | T2 undissected (arXiv 2604.18556, 2605.00649), T3 release `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF@d562806` | could beat the UD map at equal bytes | unknown until measured | high: calibration pipeline and a search loop | **MEASURE FIRST** |
| Allocation by component type rather than per-tensor reconstruction error | T2 undissected (arXiv 2609.26173, "Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity") | a cheaper rule for picking the map | unknown | low if it only informs the artifact choice | **MEASURE FIRST** |
| Treat the int8 activation quantization as the quality knob, not the weight format | T0 code (Q8_K style activations in `moe_fused_iq.cu`), T3 PTX ISA | prefill speed is already bought; this decides whether the gate passes | this is likely the largest remaining error term in the engine's own path | low: a BF16 or fp16 activation variant behind a flag | **MEASURE FIRST** |
| Fold block scales into an int8 GEMM when exponents allow (MXFP4 on INT8 units) | T4 lead: arXiv 2611.00486 abstract only, found by search, not in the research base, not dissected | would remove per-block work from the reduction | not applicable to a non-linear codebook | unknown | **REJECT** for IQ formats; **MEASURE FIRST** only for a linear 4-bit format |
| Uniform IQ4_XS across the file ("an IQ4_XS model") | T0 census (IQ4_XS is 17.1% of bytes; gate/up sits at 3.33 bpw) | none | worse: spends more bytes on gate/up and fewer on down than the published map | none | **REJECT** |

## 5. Challenges to the existing spec

**1. §7 item 3 cannot be answered as asked.** "IQ4_XS vs UD-IQ4_XS vs IQ2_XS" treats three labels as disjoint products. The artifact we run is labelled `MOSTLY_IQ4_XS` (`general.file_type = 30`) and contains IQ2_S experts, IQ3_XXS gate/up, Q5_0 and IQ4_NL down, and an IQ4_NL PLE table (T0 census above). UD-IQ4_XS already contains an IQ2_S-class allocation. The gate should name the reference by artifact identity and a tensor map, for example "the four Unsloth v3.1 shards we ship, hash and tensor census attached", not by a file-type label. This also settles the parenthetical in §5 P0 ("IQ4_XS baseline is the product quant") in one step.

**2. §5 P0's host-RAM assumption contradicts the artifact size.** P0 asks to "confirm artifact source and host RAM assumption (31 GB vs 64 GB)". This artifact is 84.5 GiB on disk, and the smallest file in the published table for this model is 72.5 GB (UD-IQ1_S), so a 31 GB host-RAM target cannot hold the sparse model at any published precision. A 31 GB target is a different model or a much smaller one, and the spec should say which, because it changes §4 and §5 P5 entirely. Our host has 64 GB.

**3. §3 item 4 names the wrong fallback.** It says to use "fp8 e4m3 (`m16n8k32`) / INT8 / IQ4_XS ggml MMQ" where NVFP4 is unavailable. In this artifact IQ4_XS is 17.1% of bytes. The kernel map should be stated per encoding, with rows for IQ3_XXS gate/up (18.2%), Q5_0 down (19.1%, currently the one encoding the fused path does not cover), and the IQ4_NL PLE gather (31.7%). As written, an implementer reading §3 would spend effort on the smallest of the four large encodings.

**4. §3 item 9 and §6 item 7 say "prefer FP32 accumulation in quality-critical paths", which does not describe where the error is.** For an IQ path the accumulator is `s32` inside the MMA and the block scales are applied in fp32 afterwards, so accumulation is not the loss term. The loss terms are the int8 activation quantization and the fp16 super-block scale of the weight format itself. The spec should name those, or the mitigation will be aimed at the wrong place.

**5. §4.2's gate is measured on a distribution the artifact was not calibrated for.** The delivered file carries an imatrix built on Unsloth's agentic, coding and multilingual set (45 chunks, 926 entries), and Unsloth's own overfitting section argues that wikitext-style KLD flatters quants tuned on wikitext (`https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`, T3). A wikitext-2 gate is still the right instrument for engine-side numerics changes, where the reference is our own BF16 path and the distribution cancels. It is the wrong instrument for comparing artifacts, which is what §7 item 3 is about. Split the two uses in §4.2.

**6. §4.1's BF16 reference needs a stated fallback for the sparse model.** A BF16 Qwen3.8-Flash-Next is 125B plus a 51.2e9-element PLE table in bf16, roughly 350 GB before activations, so the "BF16 reference path on the same 4090" of §4.1 is not buildable for the sparse target as written. Either the reference is a smaller model, or the reference is the artifact served by a different engine bit-exactly, or the gate is per-tensor reconstruction error plus task canaries. The spec should pick one; this note does not.

**Strongest challenge:** item 1, because everything else in §4 and §7 rests on it. The reference quant cannot be named by a GGUF file-type label when the label describes a sixth of the file.

## 6. Open questions and the measurements that would settle them

1. **Is decode bound by host DRAM bandwidth?** The arithmetic says about 1.05 GiB of activated expert bytes per token, and the measured 90 to 100 t/s implies 95 to 105 GB/s. Measure by running the same artifact with and without the residency split, and by plotting t/s against activated bytes per token across UD-IQ4_XS (3.33 bpw gate/up) and UD-Q3_K_XL. If the line is flat in bpw, the hypothesis is wrong.
2. **How much of the 0.0836 mean KLD comes from the IQ4_XS tensors alone?** Not answerable from the published table. The cheap local proxy is per-tensor reconstruction error against a high-precision source, which needs the source we do not have locally. Until then, treat any statement of the form "IQ4_XS costs X" as unattributed.
3. **Does the int8 activation path cost measurable KLD versus a BF16 activation path?** Same artifact, two kernels, §4.3 protocol with task canaries. This is the most valuable single GPU run in this note, because it decides whether the engine's own prefill path is the quality ceiling.
4. **Does a `.u4/.s4` MMA beat the int8 MMA on sm_89 in practice?** Microbenchmark one linear 4-bit tile with `m16n8k32` in both widths. Only worth running if a linear 4-bit reference artifact is ever adopted; it cannot help a codebook format.
5. **What is the PLE table worth?** Unsloth ships interchangeable tables (BF16, Q8_0, Q5_1, IQ4_NL) with published relative RMSE (0, 0.0053, 0.0376, 0.0757) and explicit instructions for swapping them (`set_split.py`, same model page). Running our gate with IQ4_NL against Q5_1 measures whether 9.6 GB of extra bytes buys anything we can see, which is the one place a quality-per-byte experiment is cheap.
6. **What changed between Dynamic v3.0 and v3.1?** The v3.0 file is gone locally, so the diff needs a re-download. Still open in the research-base lane `quantization`.
7. **Which revision does the 93.7 GB card row describe?** Our copy is 90.7 GB and has no MTP tensors. Until this is settled, quote the KLD row as family evidence, not as a measurement of our file.
8. **Would GSQ/RCO allocation beat the UD map at equal bytes on this model?** T2 undissected, `MEASURE FIRST`. It needs a calibration pipeline we do not have.

## 7. Sources

**Local, T0 (this session, read-only):**

- Census of `/mnt/models/Strata/models/unsloth-UD-IQ4_XS-v3.1/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001..00004-of-00004.gguf`: 1224 tensors, 176.9e9 elements, 84.5 GiB, 4.102 bpw, ten encodings, metadata as quoted.
- `~/wt/basalt-ada-lift`: `include/basalt/artifact/gguf_reader.hpp`, `include/basalt/artifact/dequant.hpp`, `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/native_mmvq.cu`, `src/models/qwen4exp/prefill/moe_fused_iq.cu`, `src/models/qwen4exp/kernels/dense_tc.cu`, `include/basalt/models/qwen4exp/decode/layer.hpp`, `docs/model-file.md`.
- `state/changes.md`: local 4090 baselines, Strata 0.2.10 warm 90 to 100 t/s and 82 to 86 t/s at 137K decode.
- `~/ninfer-4090/docs/weight-conversion.md` and `docs/llamacpp-comparison.md`.

**Official artifacts, T3:**

- llama.cpp block layout and codebook: `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h`
- llama.cpp file-type enum: `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h`
- llama.cpp quantize tool and the Llama-3.1-8B bits-per-weight table (IQ4_XS 4.4597): `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md`
- Unsloth Dynamic v3.1 method, imatrix and overfitting discussion: `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`
- Unsloth Qwen3.8-Flash-Next: BF16-referenced quant table, PLE / engram table RMSE, 4-bit minimum rationale: `https://unsloth.ai/docs/models/qwen3.8-next`
- Unsloth Dynamic 3.1 release notes: `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/dynamic-3.1-ggufs`
- PTX ISA, `mma` shape and target requirements (`.u4/.s4` `.m16n8k32` requires sm_80+; `.e4m3/.e5m2` requires sm_89+): `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html`

**Research base:**

- Lane `quantization` (living): non-uniform per-tensor GGUF is shipping; open question on the v3.0 to v3.1 diff; open question on GSQ versus imatrix at equal bits per weight.
- `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF@d562806` (T3 generated): per-tensor type map and imatrix dataset for the GSQ-RCO release.
- `2604.18556` GSQ (T2, undissected) and `2605.00649` RCO (T2, undissected).
- `2609.26173` "Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity" (T2, undissected).
- `2602.17698` ScaleBITS, `2510.16805` mixed-precision survey, `2608.24945` FAMPWQ (T2, undissected): allocation-rule candidates, not used as evidence here.
- `doc:prep4qwen/qwen38-flash-next` (T3, draft): 125B main model, 51B PLE, 4B MTP, 93.7 GB UD-IQ4_XS, no single-4090 run.
- `doc:prep4qwen/ninfer-4090-udpsendtofailed` (T3, draft): NInfer `.ninfer` semantics and 4090 numbers.

**T4 lead, not evidence:**

- arXiv 2611.00486, "Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units", abstract only, found by search, not in the research base, not dissected.
