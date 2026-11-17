> [!NOTE]
> Internal design note from 2026-11-16, kept as written apart from host, path and name scrubbing. It predates the
> engine. `SPECS.md` and `RESEARCH.md` describe what was built, and the gates in `gates/` record each step.

# BUILD SPEC — native sm_89 sparse + dense engine (BF16 next-token quality)

**Status:** build-ready specification, no code written  
**Date:** 2026-11-16  
**Fork base:** `Ex3c4Def/basalt_ada` (maintainer decision; port directly, do not rebase onto upstream `jesdga95/basalt`). Refined 2026-11-16: one-time snapshot into our own repo, no upstream afterwards, best of Strata 0.2.10 lifted in; see §9 and §10 (base is `fork_4090` merged with upstream v0.4.1)  
**Engine identity:** own breakaway name (bucket in `REVIEW_AND_STAGING.md` §1.4); not "basalt"  
**Aligned sources:** `AUDIT_LOG.md`, `../research/sm89-sparse-dense-engine-quality.md`, `RESEARCH_PREP.md`, `REVIEW_AND_STAGING.md`, `SM120_REPLACEMENT_AND_RESEARCH.md`, 10 research notes. The 2026-11-16 Qwen4 and switch refinement is in `../research/sm89-sparse-dense-engine-quality/sm89_qwen4_sparse_vs_dense_switch.md`; the IQ4_XS quality and performance lane is in `sm89_iq4xs_performance_quality_lane.md`.

This is the single source of truth after the 2026-11-16 audit. Where older docs conflict, this file wins; corrections are logged in `AUDIT_LOG.md`. The private handoff is the reading guide and work queue for the next agent; it is not a second spec.

---

## 0. The objective, stated precisely

Build a C++20/CUDA inference engine for **one NVIDIA Ada GPU (sm_89, RTX 4090 24 GB proving ground)** that runs the **Qwen4 family in both dense and sparse/MoE form**, with **Qwen3.8-27B as a temporary dense proxy** until Qwen4-27B weights land. The engine is the maintainer's default for main models and replaces every engine except llama-swap, which remains the front door. Its defining property is **quality**: the next-token distribution the engine produces must be as close as possible to a **BF16 reference** on the same hardware, measured primarily by **KL divergence (KLD)** on next-token logits plus PPL/top-1/tail metrics, with task and long-context canaries. Speed is a constraint to satisfy, not the objective. Greedy output equivalence is a diagnostic, not the target. "KLE" in project prose means this KLD-based next-token error.

**Hard scope:** Qwen sparse and Qwen dense only. No GLM, no Qwen3.6-35B-A3B or other 35B class, no DeepSeek, no other model families. Qwen3.8-27B is admitted only as the stand-in dense checkpoint while Qwen4-27B has no published weights. See §1.3.

**Replacement target:** the engine takes over the two roles it is meant to replace. Dense replaces NInfer-Qwen3.8-27B, including its 4090 decode speed and 262K context behavior. Sparse replaces Strata 0.2.10 serving QFN UD-IQ4_XS, and carries forward only what 0.2.10 proved good; nothing from 0.2.13.x, which the local A/B rejected for lower long-context decode. llama-swap stays as the front door. See §5 P6.

Why this framing: KLD/PPL can be insensitive or non-monotonic near BF16 (the "silent zone", arXiv 2606.19558), so a KLD-only claim is insufficient; task decisions can flip while aggregate scores stay flat (arXiv 2608.06564). The quality contract in §4 is therefore deliberately multi-layered.

---

## 1. Target models and verified architecture facts

Both configs verified from Hugging Face `config.json` on 2026-11-16.

### 1.1 Dense shape — Qwen4-27B is the target, Qwen3.8-27B is the temporary proxy
- **Target:** Qwen4-27B, a dense member of the Qwen4 family. As of 2026-11-16 it has no weights, config or model card; dense 27B with QSA is a T4 lead from keynote slides and kernel-code leaks, not a spec input. All dense work is built around the Qwen4 component boundary in §1.4.
- **Qualification proxy:** Qwen3.8-27B (`model_type: qwen3_5`), the only dense checkpoint available now. Its verified config below defines the dense shape the engine must run until Qwen4-27B lands. Passing this proxy is necessary but not sufficient for the Qwen4-27B claim; a new checkpoint is a config and weight-conversion change, not a new kernel port.
- `Qwen3_5ForConditionalGeneration`; 64 layers = 16 × (3 × GDN + 1 × full attention) → 48 linear + 16 full attention.
- Hidden 5120, FFN 17408. GDN: 48 value / 16 query-key heads, head dim 128, conv kernel 4, `mamba_ssm_dtype: float32`.
- Full attention: 24 Q / 4 KV heads, head dim 256, `attn_output_gate: true`, `output_gate_type: swish`, partial RoPE 0.25, MRoPE [11,11,10], rope theta 1e7.
- MTP: `mtp_num_hidden_layers: 1`, shared embeddings. Vision: ViT depth 27, hidden 1152, patch 16, merge 2.
- Context 262,144 native. This is the **dense-hybrid** path: dense FFN weights, hybrid attention; recurrent-state fidelity is the first-order correctness issue.

### 1.2 Sparse target — Qwen3.8-Flash-Next (`model_type: qwen4_exp`) — the fork's `qwen4exp` adapter
- `Qwen4ExpForConditionalGeneration`; 48 layers = 36 GDN + 12 full attention (`full_attention_interval: 4`). Hidden 2560.
- MoE: 512 experts, **10 active + 1 shared**, expert intermediate 640.
- GDN: 48 value / 16 query-key heads, head dim 128, conv kernel 4, float32 SSM state.
- Full attention: 24 Q / **2 KV** heads, head dim 256, `output_gate_type: sigmoid`, **QSA sparse attention** (`indexer_budget: 2048`, `indexer_compress_ratio: 4`, indexer 4 heads / 1 KV head, head dim 128).
- **Gated residual** (`hc_count: 4`, `hc_lowrank: 320`), **PLE n-gram embedding** at layer 2 (51B-class table, ngram_size 3, base vocab 20M, split 128, 8 heads/ngram).
- MTP: `mtp_num_hidden_layers: 1`, `hybrid: true`, QSA attention. Vision ViT depth 27. Context 262,144 native.
- This is the engine's `qwen4exp` adapter target; Qwen3-Next-80B-A3B / Qwen3-Coder-Next are architecture analogues only, not the adapter.

### 1.3 Scope boundary: Qwen only
- **Out of scope:** `deepseek41` (DeepSeek-V4.1-Flash) and any GLM, Qwen3.6-35B-A3B, Qwen3-Next-80B, or other family adapter. Do not register, ship, port, or spend qualification time on them. If the fork's `deepseek41` tree stays on disk, remove it from `src/models/builtin.cpp` so the engine cannot select it. It is not "not a first target" any more; it is not a target.
- The engine exposes exactly two model paths: Qwen dense (Qwen4-27B target, Qwen3.8-27B proxy) and Qwen sparse (Qwen4 family, Qwen3.8-Flash-Next `qwen4_exp` now).
- Dense needs a **new adapter** (fork has no dense adapter). It reuses the GDN/full-attention/ViT machinery currently inside `qwen4exp`, promoted to a shared layer per §1.6.

### 1.4 Forward compatibility: Qwen4 family (verified 2026-11-16)
- **Qwen4-Max released 2026-11-16, closed-weight, API only** (`qwen4-max` on QwenCloud; 2.4T backbone / 71B active / 328B n-gram / 1M context / native vision). No weights, no local deployment; not an engine target. Vendor param counts and benchmark tables are not spec inputs.
- **The architecture is the point.** Qwen4-Max uses the same three components Flash-Next previewed: hybrid GDN+QSA attention, gated residual, n-gram embedding. Qwen's own Flash-Next card calls it "an early preview of the architecture used in Qwen4." Therefore the verified `qwen4_exp` config in §1.2 is the best available *shape spec* for the Qwen4 family until open weights land.
- **Qwen4-27B / Qwen4-Next:** named on stage as the open-weights tier; as of 2026-11-16 no weights, no model card, no date (community rumors of Nov 17/24 are unverified). Convention suggests a dense 27B; treat that as likely, not fact. Smaller Qwen4 models are promised open-source later with no date.
- **Design consequence (no speculative code, just boundaries):** build the adapter around **components, not model ids** — GDN recurrence, QSA sparse attention (indexer budget/compress ratio), gated residual (`hc_count`/`hc_lowrank`), PLE n-gram table (`ple_layer_ids`, `ngram_size`), MTP head, ViT tower — all **config-driven** from `layer_types`, `full_attention_interval`, head counts and the same field names seen in both verified configs. A new Qwen4 checkpoint should then be a config/manifest change plus weight conversion, not a new kernel port.
- **Context:** Qwen4 is 1M native vs Qwen3.8's 262,144. The YaRN 1M frequency/temperature machinery and 1M block-table addressing from the ninfer sm_89 forks are the lever, but a 1M envelope needs depth-sensitive quality validation (§4.4) before any claim.
- **Do not bake parameter counts into code.** Blog figures for Flash-Next conflict (125B+51B+4B MTP vs 176B/6B); the HF `config.json` fields are authoritative for the implementation.
- **Refinement 2026-11-16:** the Qwen4-Max blog (primary, fetched this date) confirms 2.4T backbone / 71B active / 328B n-gram, GDN+QSA, four-branch gated residual and layer-2 n-gram embedding, API only, with smaller open models promised and no size, date or license. Qwen4-27B still has no weights, config or card; dense 27B with QSA comes from keynote slides and kernel-code leaks, so it stays a T4 lead, not a spec input.

### 1.5 Sparse versus dense: best use case on sm_89
- **Three sparse ideas, not one mode.** GDN compresses history into a fixed state and is best for long prefixes where KV growth is the limit. QSA attends to 2048 tokens per query regardless of context and is best at 64K and beyond, where attention cost and KV reads dominate. MoE keeps total capacity high with 10+1 active experts and is best when inactive experts can live in RAM or on disk and latency is secondary. PLE adds a large local n-gram lookup; gated residual widens the residual stream and allows FP8 residual state; MTP drafts tokens for verification.
- **Why QSA matters here.** Qwen's report shows QSA gains starting at 64K and 7.6x prefill / 4.9x decode at 1M for the attention module. On Ada the win is KV memory and KV read traffic, not FLOPs. That is the lever for long context on 24 GB. Research base note `doc:prep4qwen/qwen38-flash-next` (T3, draft) records the Flash-Next consumer runs; none are single 4090, and the smallest resident weight figure found is 41.73 GiB, so sparse on one 4090 means tiering.
- **sm_89 support is ours to build.** SGLang PR 36968 shows QSA sparse decode on real L20 and RTX 4090 needs classic FA2: FA4/CuTe fails at the QSA head dim 256 while classic FA2 passes parity, and the preflight fix is still open. vLLM fast QSA/GDN paths target sm_90 and sm_121. llama.cpp PR 27742 is the only consumer accuracy table and its closest hardware is sm_86. None of these is a Hopper or Blackwell kernel we can inherit on sm_89.
- **Decision rule.** Interactive, short context, tight TTFT, tight quality gate, or a host that cannot hold the sparse artifact and its tiers: use dense. Long context at 64K to 262K, agentic prefixes, repeated tool output, or capacity beyond 24 GB with RAM and NVMe available: use sparse. See §3 of the refinement note for the full table. Dense is the clean proving ground because it removes router, tier and indexer variables from KLD regressions, which is why P4 precedes P5 in §5.

### 1.6 The dense and sparse switch
- **Load time, not per token.** The fork's `src/model/registry.cpp` selects an adapter from `general.architecture` in the pack. A dense adapter registers its own name and returns `ModelDesc` with `n_expert`, `n_expert_used`, `n_expert_shared` and `n_ff_exp` at zero. `ModelSession` is already uniform, so no hot path branch is needed. `SideTableSpec` carries PLE or engram only when config has those fields.
- **The layering constraint.** `tools/check_model_layering.sh` forbids one model from including another. All GDN, QSA, gated residual, PLE and ViT code currently lives inside `src/models/qwen4exp/kernels/`. A dense adapter cannot borrow it by include. "Reuse qwen4exp machinery" in §1.3 therefore means promoting those components into the shared layer first.
- **Required work before a dense adapter.** Add one config driven `ArchSpec` from the field names shared by both verified configs (`layer_types`, `full_attention_interval`, head counts, `output_gate_type`, `hc_*`, `indexer_*`, `ple_*`/`ngram_*`, `mtp_*`, vision). Move GDN with FP32 state, full attention with partial RoPE and MRoPE, QSA indexer and sparse core, gated residual, PLE lookup, MTP head and ViT into the shared layer, parameterized by `ArchSpec` and carrying the §3 sm_89 replacements once. Then both adapters stay thin: sparse sets experts, tiers and PLE; dense zeroes experts and skips swap flags. A future Qwen4 checkpoint becomes a config and weight conversion change.
- **Quality flags stay per adapter.** Sparse needs `--adapt-swaps 0` and pinned tiers for identity runs; dense has no swaps. Acceptance bands stay separate: dense covers state, conv, RoPE and accumulation; sparse adds router order, cross-tier aggregation, indexer selection and PLE row identity.

### 1.7 Artifact source and version contract (hardlock, 2026-11-16)
- The engine accepts only Unsloth Dynamic V3.1 and later GGUFs for the Qwen dense and Qwen sparse paths, now Qwen3.8-27B and Qwen3.8-Flash-Next and later any Qwen4 checkpoint admitted by manifest. Everything else is refused at load: other community quants (HauhauCS, Dirk, OrcaSAQ, DavidAU, CyberTiel and similar), non-Unsloth IQ4_XS, `.ninfer` artifacts, and any file whose provenance cannot be verified. Maintainer decision, not a preference.
- Provenance fields verified on the live QFN v3.1 shard: `general.quantized_by = Unsloth`, `general.repo_url = https://huggingface.co/unsloth`, `quantize.imatrix.dataset = unsloth_calibration_Qwen3.8-Flash-Next.txt`, `quantize.imatrix.file = Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf`, `general.file_type = 30`, `quantize.imatrix.chunks_count = 45`, `quantize.imatrix.entries_count = 926`. The loader must check the `Unsloth` author string, the Unsloth repo URL and the `unsloth_calibration` imatrix prefix. Missing any of them is a hard refusal.
- `general.quantization_version = 2` is the GGUF format field and does not encode the Dynamic V3.1 generation. V3.1 cannot be distinguished from V3.0 by GGUF metadata alone. The gate is therefore a manifest allowlist keyed by exact Hugging Face repo, revision commit, file set and SHA-256. Future Unsloth quants are admitted by adding a reviewed manifest entry after provenance and quality checks, never by loosening the provenance rule.
- Unsloth's own Dynamic 3.1 page lists both Qwen3.8-27B and Qwen3.8-Flash-Next as V3.1 releases (`https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/dynamic-3.1-ggufs`). Research base holds T3 generated releases `unsloth/Qwen3.8-27B-GGUF@4af51b3` and `unsloth/Qwen3.8-Flash-Next-GGUF@084c631`. Pin the exact V3.1 carrying revision and hashes in P0.
- Dense consequence: the current NInfer `.ninfer` 27B artifact is not accepted by this engine. The dense target is the Unsloth V3.1 27B GGUF only. Locally only the `MTP/` head directory for that repo is present; the main quant shards must be fetched and hashed in P0 before the dense path can be qualified.
- §3.1 still applies inside the allowed files: per-encoding kernels, bytes unchanged, no requantization.

---

## 2. Hardware and toolchain contract
- **Native sm_89 only.** `CMAKE_CUDA_ARCHITECTURES` default `89`; remove the 120-only fatal guard and the "must be sm_120" GPU-select filter. No sm_86 fallback, no multi-arch in v1 (maintainer: native sm_89).
- CUDA toolkit: use what is installed on the workstation; relax the "CUDA >= 13.0" hard floor if the local toolkit is older, gating any 13-only features. Verify with `nvcc --version` before configuring. Verified 2026-11-16: CUDA 13.4.92 is installed, so the floor stands (§9.6).
- Keep CPU tests buildable (`ctest -L cpu`). GPU tests run one at a time, alone on the 4090 (no concurrent llama-swap/Strata).

---

## 3. sm_89 kernel replacement map (from `SM120_REPLACEMENT_AND_RESEARCH.md`, verified)

| # | Blackwell/sm_90+ construct | Files (basalt_ada) | sm_89 replacement |
|---|---|---|---|
| 1 | `barrier.cluster.*`, `fence.mbarrier_init.release.cluster`, `mbarrier.try_wait` suspend | `dec_gdn_rec.cu`, `hc_fused.cu`, `attn_chain_*.cuh`, `dense_tc.cu`, `qsa_mid.cu`, `dec_moe.cuh`, `qsa_select.cu`, `sampler.cu` | `__syncthreads()` / `__threadfence()`; arch-gate sm_90+ paths behind `#if __CUDA_ARCH__ >= 900` with separate TUs |
| 2 | TMA (`CUtensorMap`, `cp.async.bulk...mbarrier::complete_tx`) | `hc_fused.cu`, `mid_chain.cu` | `cp.async` (sm_80+) with double-buffered smem staging |
| 3 | `__cluster_dims__`, `cg::cluster_group` | `dec_gdn_rec.cu` | standard block-per-head launch; redistribute cluster work across blocks |
| 4 | NVFP4 / `GGML_TYPE_NVFP4` / e2m1 (15 files) | mmq instances, `moe_fp4.cu`, `expert_fp4.cpp`, CPU nvfp4 dequant | compiled out on sm_89 (Blackwell PTX only); use the per-encoding IQ MMQ table below, not one "IQ4_XS fallback"; fp8 e4m3 MMA exists on sm_89 but needs requantization, so it is not a weight path here |
| 5 | Static cooperative occupancy constants | launch sites | runtime `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; capacity functions read the same value |
| 6 | GDN cluster recurrence + TMA state transfer | `dec_gdn_rec.cu`, gdn family | standard block launch; keep **FP32 state**; ReplaySSM-style checkpoint + recent-input buffer for rollback |
| 7 | QSA TMA indexer + sm_120 tile sizes | `qsa*.cu` | fa2/fp8 plain path for QSA indexer; retune top-512 block selection for Ada occupancy |
| 8 | Gated-residual (GR) TMA/mbarrier | `hc_fused.cu` | standard smem staging, block sync |
| 9 | `mma.sync m16n8k16` tile shapes tuned for sm_120 | MoE/attention/FFN kernels | instruction exists on Ada; retune tiles for 128 SMs. For IQ weights the MMA accumulator is `s32` and block scales are applied in fp32, so accumulation is not the loss term; the loss terms are the int8 activation quantization and the fp16 weight super-block scale. Qualify those, not just the accumulator |

Port discipline from `~/<private>/reference/records/ninfer-v3-port.md`: ptxas reports only the first failure class (always rebuild `-k 0`), mbarrier cluster/PDL intrinsics must stay arch-gated, merged dispatch catalogs must be supersets, per-file numerical tolerances must be re-qualified by measurement on sm_89 (never relax blindly).

### 3.1 The quant artifact is mixed encoding (refinement 2026-11-16)
- The QFN artifact we run is `LLAMA_FTYPE_MOSTLY_IQ4_XS` (`general.file_type = 30`) with ten encodings, not a uniform IQ4_XS file. Lane read of the four shards: IQ4_NL about 38.5% of bytes (the PLE table), Q5_0 about 19.1% (down experts), IQ3_XXS about 18.2% (gate/up), IQ4_XS about 17.1%, plus Q6_K, IQ2_S, Q8_0, Q2_0, F32 and BF16. The lane's absolute byte total was corrected to the verified on-disk 93,672,359,738 bytes (87.24 GiB); see the lane note for the caveat on the per-encoding parse.
- Consequence: the MMQ dispatch must be keyed per encoding. The fork's `moe_fused_iq.cu` covers IQ gate/up and Q2_0/IQ4_NL down, but Q5_0 down experts are the uncovered row and are a large share of the file. Do not spend v1 effort on a 4-bit tensor-core path: PTX `.u4/.s4 m16n8k32` exists on sm_89, but an IQ index must be expanded through `kvalues_iq4nl` before MMA, so the codebook lookup is the work.
- Keep the artifact's bytes unchanged. Re-quantizing into a groupwise NInfer style format discards the imatrix driven per-tensor allocation, and its published A/B is against UD-Q4_K_XL, not IQ4_XS, so the quality evidence does not transfer. `.basalt` packs already copy quantized tensors bit for bit.

---

## 4. Quality contract (KLD vs BF16 next token)

### 4.1 Reference path
- A **BF16 reference path** on the same 4090: BF16 weights, FP32 GDN state, exact RoPE/mask semantics, fa2 or reference attention, no KV quantization, no speculation. This is ground truth; it may be slow.
- Every candidate path is compared against this reference on identical tokenizer, chat template, prompts/corpus, chunking, sampling/greedy settings and KV semantics.
- **Sparse BF16 fallback (refinement 2026-11-16):** a BF16 Qwen3.8-Flash-Next is roughly 125B backbone plus a 51.2e9-element PLE table, about 350 GB before activations, so the BF16 reference is not buildable on one 4090 as written. For the sparse target the reference is either the same artifact served bit-exactly by a reference engine, or per-tensor reconstruction error plus task canaries.
- **Dense BF16 fallback (final review 2026-11-16):** the dense proxy is 27B parameters, about 54 GB at BF16, which also does not fit 24 GB VRAM. "BF16 on the same 4090" needs a stated mechanism: host offload with exact math, or a declared high-precision reference served elsewhere. Pick both fallbacks in P0 and record them. Until then, every "vs BF16" phrase in this spec means the declared reference, not an assumed local BF16 run.

### 4.2 Primary gate — distributional
- Basalt's decode-path gate (keep it): wikitext-2 in 512-token chunks, first 256 prefilled, rows 256–510 scored through decode in verify windows; KLD `KL(base || test)` over full vocabulary from fp32 log-softmax summed in double, skip base terms at log-prob ≤ -16; report PPL, mean/median/p99 KLD, top-1 agreement. 20 chunks quick, 100 for numerics changes.
- Adopt Infernix/ninfer position-by-position metrics too: mean KLD `KL(reference || engine)`, p99.9 KLD tail, top-1 agreement, PPL % change; paired over ≥5–6 chunk sizes (4-bit activation paths show ~1% PPL / 0.025–0.033 KL noise from chunk/order alone).
- **Split the two uses of this gate (refinement 2026-11-16).** Wikitext-2 KLD is the right instrument for engine side numerics changes, where both legs use our own paths and the distribution cancels. It is the wrong instrument for comparing quant artifacts: the delivered UD artifact was imatrix-calibrated on agentic, coding and multilingual traffic, not wikitext, and a wikitext tuned quant can look better there than it is. Artifact comparisons must use on-distribution canaries from that same traffic family.

### 4.3 Gate policy
- **Bitwise-neutral changes must show greedy identity** (token + logit hash) on fixed prompts.
- **Numerics changes must report PPL, mean/median/p99 KLD and top-1 deltas** vs the previous build and vs the applicable reference (§4.1: BF16 for dense, the chosen sparse fallback for QFN); tail regressions and any unexplained greedy non-identity are release blockers.
- Thresholds are **empirical, calibrated from the local BF16-vs-BF16 baseline distribution** on this machine (paired comparisons, CI/repeats), not imported constants. Record hardware, driver, CUDA, commit, build flags, tokenizer, batch/chunk/graph, KV mode, seeds for falsifiability.
- Because KLD can be non-informative near BF16, add **task canaries** (1 reasoning, 1 coding/math, 1 instruction/QA) with repeat-aware comparison and **long-context NIAH/RULER** slices. Decision-sensitive probes (tool-call gating, refusals) are required wherever the artifact is sub-8-bit.

### 4.4 Per-area invariants
- **GDN:** FP32 recurrent state S; exact conv k=4 boundary/warmup/SiLU placement; alpha/beta parameterization; `norm * silu(gate)` output fusion; ReplaySSM state transactions correct under speculative rollback.
- **Attention/RoPE:** GQA expansion, attention output gate (swish on dense, sigmoid on Flash-Next), partial RoPE 0.25, MRoPE [11,11,10] sectioning/interleaved. On Flash-Next, QSA indexer top-512/r=4 and PLE n-gram lookup must match reference.
- **KV:** choose per workload; verified cosine vs FP32: int8 99.8% (≈223K ceiling), **rk4v4-e8 98.7% (433K)**, rk2v4-e8 96.2% (567K). Default to rk4v4-e8 for long context; use int8 or bf16 when KLD gate demands maximum fidelity. YaRN only with depth-sensitive validation.
- **MoE:** router logits, top-10+1 selection/normalization and aggregation order identical across resident/CPU-miss/peer tiers; byte-verify every expert slot (`verify_slot`); fence swaps at request boundaries; spill/counters observable.
- **Spec decode:** MTP/DFlash/DSpark are lossless only under correct target verification; validate greedy equivalence and seeded-sampling distribution parity vs non-spec BF16, with acceptance telemetry segmented by cache-hit/cold, depth, draft window and KV mode. vLLM FLy-style lossy verification is out of scope.
- **Serving under pressure:** bounded admission with deadlines; exact tiering (copy exact bytes) over lossy KV; cache hit/miss equivalence; determinism mode; recovery tests for kill/restart/OOM with no stale-state reuse.

---

## 5. Build phases with exit gates

- **P0 Decisions (this spec):** ratify name from bucket; pin basalt_ada commit; record the artifact by hash and tensor census, not by a GGUF file-type label; target 64 GB host RAM (verified on the workstation; the sparse artifact is 93.67 GB on disk, so a 31 GB target cannot hold it at any published precision); pick both BF16 reference fallbacks from §4.1. IQ4_XS is the maintainer's Q4 and is resolved as the first-class quant by decision 9.
- **P1 Skeleton:** CMake arch 89, remove 120 guards, toolchain check; minimal GPU smoke of a copy/attention kernel. *Gate:* configure+build on the workstation; `ctest -L cpu` green; GPU smoke alone on 4090.
- **P2 Numeric foundation:** BF16 reference path + FP32 GDN state; exact attention/RoPE/QSA/PLE for both adapters; kernel parity tests. *Gate:* BF16 reference passes determinism; first KLD/PPL/top-1 numbers recorded vs reference.
- **P3 sm_89 hot paths:** apply §3 replacement map; attention/GEMM/MMQ/MoE on sm_89; memory/KV/graph limits. *Gate:* kernel parity green on 4090; KLD gates from §4.2; VRAM/RAM accounting logged.
- **P4 Dense end-to-end:** new dense adapter reusing qwen4exp GDN/ViT; loader, runtime, OpenAI-compatible serve. Keep the component/config boundary of §1.4 so Qwen4-27B is a config + conversion change. Qualify on the Qwen3.8-27B Unsloth V3.1 GGUF proxy. *Gate:* coherent generation; greedy identity vs the declared dense reference (§4.1); TTFT and decode baseline measured against current NInfer-Qwen3.8-27B on the same 4090, same artifact where possible. The engine must match NInfer's decode behavior before it can replace it.
- **P5 Sparse end-to-end:** Flash-Next router/top-10+1, PLE n-gram tiering, QSA, gated residual, VRAM/RAM/disk expert residency, dense+sparse shared runtime. *Gate:* correctness vs the §4.1 sparse reference chosen in P0; residency/eviction stability; KLD + task canaries under tier pressure.
- **P6 Productize:** serving/ops, llama-swap/Berth wiring as the sole front door, validation matrix automated, docs, and a migration plan off both Strata (sparse) and NInfer (dense). v0 supports only the Qwen dense and Qwen sparse paths in this spec. *Gate:* can serve the maintainer's main models with llama-swap in front, with parity evidence against the two engines it replaces.

---

## 6. Decisions resolved (were open questions)
1. **Base:** fork `Ex3c4Def/basalt_ada` directly; upstream cherry-picks optional. *(maintainer)*
2. **Native target:** sm_89 only; no sm_86/multi-arch in v1. *(maintainer: "native sm_89 only")*
3. **Quality metric:** KLD next-token vs BF16 is primary; greedy is a diagnostic. *(this spec)*
4. **Dense path:** new dense adapter reusing promoted qwen4exp components. Qwen4-27B is the intended main model but has no weights as of 2026-11-16, so qualification runs on Qwen3.8-27B as a temporary proxy. *(this spec, updated 2026-11-16 to match §1.1 and decision 10)*
5. **Sparse model:** Qwen3.8-Flash-Next (qwen4_exp); verified config. *(audit correction E1)*
6. **KV default:** rk4v4-e8 for long context, int8/bf16 for fidelity-critical work. *(verified data)*
7. **Precision:** FP32 GDN state. Accumulation precision is qualified per path; for IQ weights the MMA accumulator is already `s32` with fp32 scale application, and the measured loss terms are int8 activations and the fp16 weight scale (§3 row 9 and §3.1). fp8/INT8/IQ4_XS only with gates. *(this spec, tightened 2026-11-16)*
8. **Dense/sparse switch:** load time adapter selection, not a runtime mode. A dense adapter is a zero-expert `ModelDesc`; shared architecture components are promoted out of `qwen4exp` into a config driven shared layer before the dense adapter is written. *(refinement 2026-11-16, code verified in `~/wt/basalt-ada-lift`)*
9. **First-class quant:** IQ4_XS class, served by ingesting the UD artifact unchanged and dispatching per encoding. No re-quantization to a groupwise format and no own packed layout in v1; a 4-bit tensor-core weight path is rejected because the IQ index needs codebook expansion before MMA. *(refinement 2026-11-16, IQ4_XS lane)*
10. **Engine scope and role:** Qwen dense and Qwen sparse only, tuned for the Qwen4 family. Qwen4-27B is the intended main model; Qwen3.8-27B is a temporary dense proxy. No GLM, 35B class, DeepSeek or other families. The engine replaces NInfer and Strata for main-model serving; llama-swap stays as the front door. *(maintainer, 2026-11-16)*
11. **Artifact source:** Unsloth Dynamic V3.1 and later GGUFs only, enforced by manifest allowlist plus provenance fields. No community quants, no `.ninfer`, no unverifiable provenance. *(maintainer, 2026-11-16)*

## 7. Still open (maintainer/architect)
1. Engine name — pick and verify GitHub/PyPI/trademark from the unique-name bucket.
2. Pin the exact `basalt_ada` commit hash for the fork.
3. BF16 reference fallback for both paths: sparse (bit-exact artifact via reference engine, or per-tensor reconstruction plus canaries) and dense (host-offloaded BF16, or a declared high-precision reference). (§4.1)
4. v0 API surface: streaming, tool calls, vision, grammar, multi-slot concurrency?
5. KLD/task acceptance bands — calibrated from the local BF16 baseline, not set here by design.
6. Whether decode is host-DRAM bound is a derived hypothesis (~1.05 GiB activated expert bytes per token at 90 to 100 t/s implies ~95 to 105 GB/s), not a measurement. Settle with a residency A/B and a bytes-per-token sweep before designing around it.

Items removed as resolved on 2026-11-16: reference product quant (decision 9 plus the manifest identity rule in §1.7, not a GGUF file-type label), host RAM target (64 GB, verified on the workstation).

## 8. Attribution / license
- Fork is MIT (Jesús Gulfo); Strata MIT; basalt ggml MIT; NInfer Apache-2.0; model weights carry their own licenses. Preserve LICENSE/THIRD_PARTY_LICENSES and source headers.
- `CONTRIBUTING.md` asks for `Co-Authored-By` on ported work, but maintainer forbids AI attribution trailers. **Rule:** preserve original authors in notices/headers; emit **no** `Co-Authored-By` git trailers.

---

## 9. Refinement 2026-11-16 (review 2): own repo, no upstream, Strata lifts, measured targets

Maintainer wording, same day: "taking the best from strata + basalt base and we make our own forked code and engine for
sm_89. I don't want any upstream engine for my repo at all." "A system that dynamically does NOT change." "Best KLE,
DECODE + PREFILL possible only on SM_89." This section tightens §§1 to 8 and wins where it differs. Scope, the Unsloth
hardlock, bytes-unchanged and the phase order are unchanged.

### 9.1 Repo rule: a one-time snapshot, then no upstream
- The engine lives in its own new repository, created empty. It is not a GitHub fork and has no fork relationship.
- One import commit from the pinned Basalt snapshot (§9.2). After it: no `upstream` remote, no remote to any Basalt or
  Strata repo, no submodule or vendored updater pointing at an engine, no `update.sh`, no cherry-pick stream. This
  replaces "upstream cherry-picks optional" in §6.1.
- Later ideas from Basalt, Strata, NInfer or papers come in by reading and reimplementing. Each one gets a row in a
  `LIFTS.md` ledger: source project, version, file, licence, what was taken, the A/B that justified it.
- MIT still requires the copyright notices. Keep `LICENSE`, `THIRD_PARTY_LICENSES` and source headers (§8). That is
  attribution, not an upstream.

### 9.2 Lineage and the snapshot to pin (verified in `~/wt/basalt-ada-lift`, `~/strata/releases/0.2.10`)
- Lineage: `Niko1221/Strata` (MIT) -> `jesdga95/basalt` (MIT) -> `Ex3c4Def/basalt_ada`. Basalt's own `CONTRIBUTING.md`
  says it is based on Strata.
- The local `basalt_ada` checkout is HEAD `499fad2`, `project(basalt VERSION 0.3.0)`, about 128K lines of C++/CUDA under
  `src` and `include`. Its `CMakeLists.txt` still defaults to arch 120, fails on anything but 12.0 and builds
  `model_qwen4exp_fp4` as `120a-real`. No Ada port work is visible in it. Plan the full sm_89 bring-up; expect nothing
  pre-ported. **Corrected by §10.1: this is true of `main` only; branch `fork_4090` holds a working Ada port.** The checkout shows one commit, so it is probably shallow.
- Upstream Basalt is at 0.3.3 in a third-party report dated 2026-11-14 (T4). P0 item: take a full-history clone of both,
  diff `basalt_ada` against `jesdga95/basalt`, and record what the fork adds. Then pin one hash. This is the only time
  the question is asked, because after the import there is no upstream.
- Strata 0.2.10 on this host is 252K lines of C++/CUDA plus 42K of Python serve, built locally for sm_89. Its three
  provenance notes disagree: `COMMIT` says `af6224f3` (also the head of `deploy/the workstation-0210` in `~/strata-dev`),
  `RELEASE.txt` says head `eff3155` on tag `db0500a`, the eval README says tag `ffa80d73`. Use `af6224f3` as the
  reference identity and note the disagreement in `AUDIT_LOG.md`.

### 9.3 What to lift from Strata 0.2.10
Strata is the only engine that serves this exact artifact on this exact card today. That makes it more than a behaviour
reference.

| Lift | Evidence | Where it lands |
|---|---|---|
| sm_89 ancestors of the §3 kernels. Basalt descends from Strata, and Strata still carries the pre-Blackwell paths (`__CUDA_ARCH__ >= 800` guards, `src/prefill/moe_fused_iq.cu`, `mmq_kq_lib.cu`, `gdn_tc.cuh`, `moe_mmq.cu`). | Compiled for sm_89 and serving daily | P3. For every §3 row, read the Strata ancestor before writing a replacement, and use the Strata build as the parity oracle for that kernel. |
| The Q5_0 down-expert path. §3.1 calls it the uncovered row in Basalt. Strata serves the ten-encoding artifact, so a path exists there. | Inferred from the artifact being served; locate the code in P0 | P3 |
| GGUF shards read in place (`src/core/expert_source.cpp`, `gguf_mode`, "no experts.bin"). | Code comments at lines 1470 and 5800 | P0 decision, §9.7. Avoids a second 87 GiB copy as a `.basalt` pack. |
| DFlash2 drafter with prompt lookup. | Local A/B 2026-11-04: +6 to +16% decode on code and agent turns against MTP, rewrite turns 162 t/s with lookup | P5, one drafter fixed per profile |
| The four upstream fixes in the local release: #5183, #5218, #5249 (serve), #5108 (prefill MMQ stream-k fixup room). | `RELEASE.txt` | P4 and P6 serve, P3 MMQ |
| The 262K operating profile and its limits (about 15 to 16 GB VRAM, about 21 GiB pinned RAM, host-RAM sensitivity). | `guide/strata-and-ninfer.md` | P5 budgets |

Do not lift: the multi-vendor trees (HIP, SYCL, Metal, Spark), Windows and installer code, `dsv41`, the `q36` engine
under `src/qwen35`, name-based feature selection (it silently ran MTP for a DFlash2 request, 2026-11-04), anything from
0.2.13.x.

### 9.4 Speed targets: local floors, not Blackwell numbers
- Basalt's published lead is on sm_120 with NVFP4 experts on FP4 tensor cores. Its own precision page measures the MMQ
  path 17 to 19% slower on prefill than FP4 (16K: 6,456 against 7,807 tok/s). NVFP4 is compiled out on sm_89 (§3 row 4),
  so MMQ is our prefill path and a 5090 number is not a target. No Basalt sm_89 number exists anywhere. **Corrected by §10.1: one exists on `fork_4090`, and NVFP4 runs on 8.9 through MMQ.**
- Speed is now a gate with numbers, not only "a constraint" (§0). Floors are the two engines being replaced, measured on
  this card:

| Path | Floor today | Source |
|---|---|---|
| Sparse decode, QFN UD-IQ4_XS v3.1 | 82.5 t/s short with MTP, 90.9 with DFlash2; 71.8 at 36K; 82.2 at 137K | Strata 0.2.10, 2026-11-04, q4_0 KV |
| Sparse prefill | about 2,410 t/s at 36K, about 2,870 at 137K | same |
| Dense decode, Qwen3.8-27B | about 130 t/s (131.2 with MTP at 70.8% acceptance) | NInfer, `.ninfer` artifact |
| Dense prefill | about 3,500 t/s at 20 to 25K, INT8 | NInfer, 2026-10-03 |

- P0 re-measures these floors with the protocol in §9.6 so every later phase compares like with like. P5 and P4 pass only
  at or above the floor at every measured context. Stretch targets are set after P3 from the measured ceilings (VRAM
  bandwidth for dense, activated expert bytes per token over PCIe for sparse, §7.6).
- Dense risk, stated plainly: NInfer reaches its number with its own artifact layout. Whether Unsloth GGUF bytes served
  unchanged can match it is unproven. If P4 cannot reach the floor, the maintainer chooses between decision 9 (bytes
  unchanged) and the NInfer bar. NInfer stays in service until P4 passes.

### 9.5 KLE: where the error is and the order to remove it
- **Reference mechanism (closes most of §7.3).** Basalt's own method works here: Unsloth's BF16 GGUF through
  `llama-perplexity --kl-divergence-base`, 100 chunks of 512 on wikitext-2, producing one frozen `.kld` base file. Run
  it with the lab's pinned llama.cpp. Dense: the 27B BF16 GGUF is about 54 GB and runs with partial offload on the workstation
  (24 GB VRAM plus 62 GiB RAM), slowly, once. Sparse: the same command with `--n-cpu-moe 99` needs the eight BF16
  shards, about 350 GB, readable from disk. That needs storage this host does not have free (§9.7), or a borrowed
  machine for one run. The base file is hashed and stored; the reference never moves afterwards.
- A quantized base is not a reference. Basalt measures 0.003 to 0.010 KLD of extra error against a Q8_0 base.
- Add canary bases the same way from on-distribution text (agentic, coding), per §4.2.
- **Engine budget.** On Flash-Next, Basalt measures all engine roundings together at about 0.009 KLD, 15% of the total
  against BF16 (0.0531 with every rounding off, 0.0621 default). The other 85% is the artifact. So the engine goal is
  concrete: reach the "all roundings off" number at production speed, and never add to it.
- **KV is the largest single engine term.** Basalt's paired dKLD: int8 +0.0000 [-0.0010, +0.0010], k8v4 +0.0056,
  q4_0 +0.0093. Strata here runs q4_0, which costs about as much as every other engine rounding combined. Derived from
  the configs, to be measured: QFN KV is 12 layers x 2 KV heads x 256 x 2 = 12.3 KB per token at int8, 3.2 GB at 262K,
  so int8 fits for sparse. Dense is 16 x 4 x 256 x 2 = 32.8 KB per token, 8.6 GB at 262K, which matches the 223K int8
  ceiling in §4.4 and needs a 4-bit format for 262K.
- **Proposed change to decision 6** (maintainer to confirm): sparse default int8 at every context that fits; dense int8
  to its ceiling and a 4-bit format only for the 262K profile. `rk4v4-e8` is a NInfer format and is not in the Basalt
  tree, so it is a port item; qualify it by paired dKLD, not by cosine.
- **Other terms, in Basalt's measured order:** Q6_K dense projections about 0.006 (artifact, not ours to change), Q6_K
  head +0.0022, int8 activations into the head about 0.0001. Build the same ledger on sm_89 in P3: one switch per row,
  paired dKLD with a 95% CI and top-1 points.
- **Engine-versus-llama.cpp gate.** On the same Unsloth weights, our KLD against the BF16 base must be at or below the
  pinned llama.cpp's. Basalt shows this is reachable (0.0200 against 0.0325 at Q8_0). llama-swap stays, so the
  comparison engine is always on the host.

### 9.6 Frozen engine: it does not change unless a release changes it
- One arch (`89-real`), one toolchain, recorded per release. Verified on the workstation 2026-11-16: CUDA 13.4.92, driver
  615.78.08, RTX 4090 24,564 MiB, 62 GiB RAM. The CUDA 13 floor in §2 needs no relaxing.
- No `auto` at run time. Expert cache size, prefill mode, drafter, draft window, KV format, residency plan, worker count
  and affinity are resolved once per model into a checked-in profile keyed by manifest id. The engine prints a hash of
  build + profile + manifest at start. Same hash, same behaviour.
- No behaviour keyed on a model name. Everything keys on the manifest.
- Expert swaps copy exact bytes, so residency never changes output, only speed. The swap policy and its parameters are
  fixed in the profile. Whether swaps are on at all is an A/B result, not a default.
- A release is engine commit + profile hash + manifest hash + gate report. The gate fails a release that is worse than
  the previous one on mean or p99 KLD beyond its CI, or on decode or prefill beyond run-to-run range. A regression
  cannot ship quietly, which is the complaint about recent Strata releases.
- Measurement protocol for every speed number: at least 5 hot rounds, report the range, end-to-end wall time, void a
  round whose cold start is polluted by leftover VRAM, single lane unless stated.
- Host tuning is part of the profile, not discovered at run time: workers pinned to the eight P-cores of the i9-14900K,
  CPU code built for AVX2 and AVX-VNNI (the chip has no AVX-512), huge pages for the pinned expert pool, direct reads
  for the PLE table on NVMe. Each is an A/B in P3 or P5.

### 9.7 P0 blocker: disk
- `/mnt/models` is 938 GB with 24 GB free (98%), measured 2026-11-16. The plan needs the 27B Q4 shards (about 15 to
  18 GB), the 27B BF16 GGUF for the reference (about 54 GB), and, if `.basalt` packs stay, a second copy of the 87 GiB
  QFN artifact. None of that fits. A nearly full SSD is also the PLE read path.
- Recommendation: lift Strata's GGUF-in-place reading so the artifact exists once, free at least 150 GB before P0
  fetches (the maintainer picks what goes), and keep 15% of the drive free as a standing rule.
- Fetch two dense candidates if space allows, UD-IQ4_XS and UD-Q4_K_XL, and choose by KLD per GB of VRAM with the 262K
  profile in mind.

### 9.8 Additions to the phases in §5
- **P0** adds: full-history diff and pin (§9.2), the disk plan (§9.7), the two `.kld` bases or a dated plan for the
  sparse one (§9.5), the floor re-measurement (§9.4), the location of Strata's Q5_0 path (§9.3), the decision-6
  proposal, and the Inbox ticket.
- **P1** adds: the empty repo and single import commit (§9.1), `deepseek41` and every sm_120-only target removed at
  import, `LIFTS.md` started.
- **P2** adds: the "all roundings off" switch set as the reference mode; the gate tool reading a `.kld` base.
- **P3** adds: Strata-ancestor parity per §3 row; the rounding ledger; the profile and hash mechanism (§9.6).
- **P4** gate adds: the dense floor in §9.4 and the llama.cpp gate in §9.5.
- **P5** gate adds: the sparse floors in §9.4 at short, 36K and 137K; int8 KV at 262K measured for VRAM and dKLD.
- **P6** adds: the release gate of §9.6 automated; Strata and NInfer kept installed as rollback until two releases pass.
- **Qwen4 day:** a new checkpoint is a manifest entry, a config, a `.kld` base and a profile. Expect new kernel work
  only for the 1M context envelope (§1.4).

### 9.9 New material found 2026-11-16 (all T4 unless stated; none is sm_89)
- `github.com/lifeidle/qwen3.8-flash-next-5090-24gb-strata-vs-basalt` (2026-11-14): Basalt 0.3.3 against Strata 0.1.55
  on a 24 GB 5090 laptop, UD-IQ1_S: 96.1 -> 143.3 t/s single lane, 130K prefill about 2,700 -> 3,716 t/s. Old Strata,
  Windows, different quant, Blackwell. Useful for its measurement lessons (§9.6) and the KV table.
- `thomaswildetech.com/blog/2026-11-11/two-engines-one-model-basalt-versus-strata-on-the-5090/`: Basalt 0.3.2 against
  Strata 0.2.13 with a GGUF-to-pack conversion. Not read yet; queue it with `rkb lead`.
- `basalt.jgulfo.com/showcase/engine-comparison/`: seven engines on one 5090 with NVFP4, October 2026. Vendor page.
- Basalt `docs/precision/qwen38-flash-next.md` and `quality.md` in the local checkout: the source of the KLD numbers in
  §9.5. Treat as T3 for method, re-measure every value on sm_89.
- Qwen4 open weights: a search on 2026-11-16 found no Qwen4-27B or Flash-Next successor weights. §1.4 stands.
- The research base has no note on Strata or Basalt. File the three links above as leads.

---

## 10. Refinement 2026-11-16 (review 3): the real base, no DeepSeek, metrics-only web UI

Maintainer, same day: "Go off the latest basalt base then (v0.4.1). If that fork has no ada code, then lets start with
the most recent version so we lose nothing." "I do not want anything for Deepseek V4.1 Flash." "The webUI only needs a
metrics + model stats page ... wildly displayed different so there is no resemblance, but the level of detail and
information should be presented all the same." This section wins over §§1 to 9 where it differs.

### 10.1 Correction: the fork does have an Ada port
Review 2 (§9.2, §9.4) looked only at `basalt_ada` `main` and said no Ada work was visible. That was wrong. The work is
on branch **`fork_4090`**. Verified from a full-history clone in `~/wt/basalt-upstream-lift` and from the branch's own
`ADA_PORT_NOTES.md`:

| Fact | Value |
|---|---|
| `fork_4090` head | `24362905`, 2026-11-15, version 0.4.0, 32 commits of its own, one author |
| Upstream latest | tag `v0.4.1` = `d3b75742`, 2026-11-16 |
| Merge base | `d7347724` (upstream 0.4.0+4). Upstream has 44 commits the fork lacks |
| Trial merge of the two | two conflicts: `CMakeLists.txt`, `src/prefill/fp8_gemm.cu` |
| Build | `-DCMAKE_CUDA_ARCHITECTURES=89` supported; sm_90+ instructions are guarded and a runtime predicate picks the fallback |
| Ada kernels | `k_mid_ada` (GDN chain), `k_hc_ada` and `hc_fused_ada` (gated-residual reads), `dm_plan_ada` / `dm_ring_ada` (route), flat `gdn_gating_proj`, f16-mma forms of the FP8 and FP4 dense products |
| Tests on a 4090 | GPU suite 116 of 124 pass, 6 skip, 2 fail (one Windows-only, one tolerance); parity tests for IQ1_S to Q8_0, the K-quants and Q2_0 pass |
| Run end to end | only the NVFP4 Flash-Next pack. Development figure at 450 W: decode 170.2 -> 180.1 t/s on the community bench after the Ada kernels |
| Not run | any Unsloth IQ4_XS pack, any 27B pack, any Linux build. The port was developed on Windows |

Consequences for earlier sections:
- **§3 is mostly done, not to do.** Rows 1 to 3 and 5 to 8 have implementations on `fork_4090`. §3 becomes a
  verification list: build each on Linux, run its parity test, and still use Strata 0.2.10 as a second oracle (§9.3).
- **§3 row 4 is wrong as written.** NVFP4 is not compiled out on this code: NVFP4 experts run on 8.9 through MMQ with
  software decode. It stays unused here only because the Unsloth hardlock (§1.7) excludes NVFP4 packs.
- **§9.4's "no Basalt sm_89 number exists" is wrong.** One exists, above. It is a different quant on a different bench
  from Strata's, so it is not comparable to the floors, which stand.
- **The dense adapter already exists upstream since 0.4.0** (`src/dense`, `src/models/qwen35`, "the shared dense
  stack", GDN and attention kernels shared with Flash-Next). The promotion work in §1.6 and "new dense adapter" in §1.3,
  decision 4 and P4 are done by upstream. What is left for dense is in §10.3.

### 10.2 The base: both, merged once, so nothing is lost
- Import = `fork_4090` at `24362905` merged with upstream `v0.4.1` at `d3b75742`, conflicts resolved, as the single
  import commit of §9.1. Record both hashes and the merge base in `LIFTS.md`. This answers §7.2 and review 2's pin item.
- The fork had Ada code, so the "start from the most recent version" fallback is not needed; taking both is strictly
  more than either.
- The 44 upstream commits are mostly the two-card layer split for the 27B, serve fixes, the cgroup-aware RAM plan (#339
  and two `platform` commits) and DeepSeek tiers. Keep the serve and RAM-plan fixes; the rest falls to §10.4.
- After the import there is no upstream (§9.1). Later Basalt releases are read, not merged.

### 10.3 Dense: the stack exists, the artifact path does not
- Basalt's only 27B packs come from QUASAR's QAT NVFP4 checkpoint or NVIDIA's NVFP4/FP8 checkpoint, 16.0 GiB on the
  card with the block drafter, 14.0 without. Neither is an Unsloth GGUF, so both are refused by §1.7. They are the only
  dense artifacts proven on this code. Reopening that is the maintainer's call; this plan does not.
- The dense work is therefore: Unsloth GGUF encodings (K-quants and IQ) as weights of the existing dense stack, for
  decode windows and prompt chunks. P0 check: whether `convert.py` accepts a `qwen35` GGUF at all today.
- KV on the dense stack, from upstream's 27B page: fp16 is 64 KiB per position, int8 is 33 KiB (matches the 32.8 KB
  derived in §9.5). 262,144 positions at int8 is 8.6 GiB. With about 16 GiB of weights that does not fit 24 GB.
  Upstream's own answer is a 32 GB card or `--kv-resident`, which streams far positions from RAM and slows every token.
  So dense at 262K on the 4090 needs one of: a smaller Unsloth quant, a 4-bit KV port (§9.5), or accepting the
  streaming cost. Measure all three in P4. NInfer's 262K at about 130 t/s is the bar.
- The dense drafter upstream is DFlash 2 (`z-lab/Qwen3.8-27B-DFlash2`), not an Unsloth file. Maintainer decision: whether
  a drafter counts under the Unsloth rule. A draft is always verified by the target model, so it cannot change output.
  Unsloth's repo carries an MTP head (already on disk), which is the in-rule alternative.

### 10.4 Remove at import: DeepSeek and everything the host cannot use
- **DeepSeek-V4.1-Flash, all of it** (maintainer): `src/models/deepseek41`, its headers, the DSpark drafter, the Engram
  table code, `ds41_*` tests and frozen hashes, serve dialect, docs, profiles. 102 files, about 1.26 MB of a 15.2 MB
  tree. This replaces "remove it from `builtin.cpp`" in §1.3: the code goes, not just the registration.
- **All sm_120 and mixed-arch code** (already decided, §2): the `120a-real` targets, FP4 tensor-core kernels, the
  cluster, TMA and PDL variants, the `"120;89"` build.
- **Windows build and platform code.** The engine is Linux on the workstation only.
- **Recommended, maintainer to confirm: multi-GPU.** Peer expert tier, layer split, prompt split, two-card tests: about
  32 files and 0.36 MB. The host has one card and the fork's notes say none of it was run end to end on Ada.
- **Recommended: the in-engine model router.** llama-swap is the front door and loads one model per process.
- **Gate for every removal:** greedy output and the quality gate's rows byte-identical before and after, with
  `--expert-cache` fixed. The fork's notes show a different binary size changes the slot count and nearly every row
  (6,491 against 6,515 slots, KLD 0.0620 against 0.0619), so an unfixed comparison reads as a false regression.

### 10.5 BF16 reference: published bases exist
- Upstream publishes its BF16 KLD bases as the Hugging Face dataset `jesdga/basalt-kld-bases`, with the Flash-Next base
  and, since 0.4.0, the 27B base. This removes the 350 GB problem in §9.5.
- P0: download, record SHA-256, and store beside the manifests. Regenerate the 27B base once locally from Unsloth's
  BF16 GGUF with the pinned llama.cpp (§9.5) and compare, so the reference is verified rather than trusted.
  The Flash-Next base cannot be regenerated here; record it as T3 and say so in every report that uses it.

### 10.6 Web UI: metrics and model stats only, our own design
- **Scope:** one read-only page. No chat, no prompt box, no API-key entry, no model switching, nothing from Strata's
  web UI. The page reads one JSON status endpoint. Add a Prometheus `/metrics` endpoint with the same values, so the
  lab's Grafana and a Kuma monitor read the engine without the page.
- **Information parity with Basalt's page** (inventory taken from `serve/web/` at v0.4.1). The page must show all of:

| Group | Fields |
|---|---|
| Model | name, state, load progress, context used and size, slots, max tokens |
| Host | hostname, CPU name, RAM used and total, disk read MB |
| GPU | memory used and total, power limit, PCIe generation (current and max) and width |
| Weights | GPU and host weight bytes; for sparse: expert slots, expert total, hit rate, VRAM / PCIe / SSD share |
| Decode | tok/s now and over time (chart), per-slot tok/s, drafts offered, draft length |
| Prefill | tok/s now, mean and over time (chart), last prefill, cache hit, restored ms |
| Requests | per-request log: prompt tokens, output tokens, TTFT, prompt ms, elapsed, totals |

- **Ours in addition:** the build + profile + manifest hash (§9.6), manifest id and artifact revision, KV format, the
  running release's gate numbers (KLD mean and p99, top-1, decode and prefill floors).
- **No resemblance:** written from scratch. Nothing from `serve/web/` or `serve/frontend.py` is copied: no markup, CSS,
  JS, fonts choice, mark, lockup or layout. Different structure, type, colour and chart style. Acceptance is a
  field-by-field parity check against the table plus a side-by-side screenshot review by the maintainer.
- **Binding:** loopback, published tailnet-only like every other lab service.
- **When:** P6. The status endpoint and `/metrics` land in P4 with the first serve, because the gates need the numbers.

### 10.7 Phase changes on top of §5 and §9.8
- **P0:** pin `24362905` + `d3b75742`; download and hash the KLD bases; check `convert.py` for a `qwen35` GGUF path;
  disk (§9.7) still blocks, since a Flash-Next `.basalt` pack is a second 87 GiB copy until GGUF-in-place is lifted.
- **P1:** the merged import; first Linux `89-real` build with `-k 0`; run the fork's GPU suite alone on the card and
  compare with its 116 of 124; then the §10.4 removals under their gate.
- **P2 and P3 shrink.** The reference mode, gate tool and most sm_89 kernels exist. The work is the rounding ledger,
  the frozen profile (§9.6), the Q5_0 row, and the first end-to-end run of the Unsloth UD-IQ4_XS artifact.
- **Order change:** sparse now runs before dense. The sparse path exists and has run on a 4090; the first thing worth
  knowing is its decode and prefill on our artifact against the Strata floors. Dense follows as §10.3. This reverses
  the P4-before-P5 order of §5, whose reason (no dense code, no sparse build) no longer holds.

---

## 11. Maintainer decisions 2026-11-16 (review 3 follow-up)

### 11.1 Dense weights: lossless repack of the Unsloth GGUF, not NInfer-style requantization
- **What NInfer actually does** (verified, `~/ninfer-4090/docs/weight-conversion.md`): it does not convert the GGUF.
  It requantizes the BF16 safetensors checkpoint itself with `grouped_absmax` (`q4_g64_fp16` and siblings: one FP16
  scale per 64 values, plain absmax, no imatrix) into a layout its kernels read fast. It can also `import_encoded`
  NVFP4/FP8 words unchanged. Its speed comes from the layout; its rounding is simpler than Unsloth's.
- **Decision: repack, don't requantize.** The converter reads the Unsloth GGUF and writes the same codes and the same
  scales in a layout chosen for the sm_89 kernels (tile order, interleave, alignment, scales split out). Values are
  unchanged. That gives the speed reason NInfer converts and the precision of Unsloth's imatrix-driven per-tensor
  allocation, which a plain absmax requantization throws away. This is how `.basalt` packs already treat quantized
  tensors, and it keeps decision 9 and §1.7 intact.
- **Proof of lossless:** for every tensor, dequantized values of the pack equal the GGUF's bit for bit (hash per
  tensor in the manifest). A repack that fails this is a requantization and is refused.
- **Requantizing stays out** unless a measured challenger beats it: if P4 misses the NInfer floor with the repack, the
  next step is a measured A/B, not a quiet switch.
- **Fit at 262K, two configurations to A/B** (derived, measure in P4). The 24,564 MiB card, embedding in pinned RAM as
  upstream does (2.4 GiB off the card), int8 KV 33 KiB/position = 8.25 GiB at 262,144, K8V4 about 24.75 KiB = 6.2 GiB:

  | Config | Weights on card | KV at 262K | Fits |
  |---|---|---|---|
  | UD-Q4_K_XL (16.68 GiB file) + K8V4 KV | about 14.3 GiB | 6.2 GiB | yes, about 3.5 GiB left for activations, graphs and drafter |
  | UD-IQ4_XS (size to confirm in P0) + int8 KV | about 12 GiB if the file is about 14.5 GiB | 8.25 GiB | likely |
  | UD-Q4_K_XL + int8 KV | about 14.3 GiB | 8.25 GiB | no headroom for the drafter |

  Choose by whole-system KLD against the BF16 base at long context (KV error grows with depth) plus the decode floor.
  Weight error and KV error trade off here, so neither is chosen in advance.

### 11.2 Drafter: A/B DFlash2 against MTP, on both paths
- Both run in their own arms; the drafter cannot change output, so the A/B is speed only, plus a greedy-identity check
  in each arm. Same protocol as the 2026-11-04 Strata A/B (`evals/2026-11-04/strata-0210-drafting/`): short,
  4K, 16K, 36K, 137K, code, agent edit, prose, rewrite. Remember DFlash2 turned prompt lookup off and lost 13% on the
  rewrite workload.
- Sources: sparse MTP from the Unsloth artifact; sparse DFlash2 as Strata 0.2.10 used it; dense MTP from Unsloth's
  27B `MTP/` directory (on disk); dense DFlash2 `z-lab/Qwen3.8-27B-DFlash2` (upstream's choice). The Unsloth-only rule
  of §1.7 covers the target model's weights; drafters are admitted by manifest the same way, because their output is
  always verified.
- The winner is fixed per profile (§9.6), per workload class if the A/B shows a split. No run-time drafter choice.

### 11.3 Multi-GPU removed (confirms §10.4)
- One card, sm_89 only. Remove at import: peer expert tier, layer split, prompt split, GDN and head splits, staged
  copies between cards, the second-card state save/restore, two-card tests, and the upstream 0.4.1 commits that add the
  27B layer split. Same byte-identity gate as §10.4. The in-engine model router goes too.

### 11.4 Engine name: shortlist, checked 2026-11-16
Unrelated to Qwen, Basalt, Strata or NVIDIA. Checked: PyPI, crates.io, npm (404 = free), GitHub repository-name
search, Hugging Face model search, a web search for software of the same name. **Not checked: a formal USPTO/EUIPO
trademark search.** Run one before any public release.

| Name | PyPI | crates | npm | GitHub repos named so | Software conflict found | Meaning |
|---|---|---|---|---|---|---|
| **Halocline** | free | free | free | 31, none an engine | none | The sharp layer in deep water where salinity changes. Fits the lab's deep-water names (The workstation, HADAL) |
| **Komatiite** | free | free | free | one small C# templating repo | that repo only | The hottest, most fluid lava the Earth has erupted: an ultramafic rock hotter than basalt |
| **Anorthite** | free | free | free | 2 | none | The calcium feldspar that crystallizes out of basalt melt |
| Pycnocline | free | free | free | 17 | none | The density layer of the deep ocean. Hard to say |
| Tholeiite | free | free | not checked | 0 | none | A basalt family. Hard to spell and say |

Rejected on conflicts: Gabbro, Tephra, Kimberlite, Scoria, Peridot, Xenolith, Caldera, Spinel (taken on PyPI and
crates), Olivine and Augite (npm taken), Lapilli (crates), Fumarole (crates), Thermocline (PyPI), Nekton and Ophiolite
(PyPI), Benthic and Bathyal (inference projects on GitHub), Basanite (too close to Basalt).

Recommendation: **Halocline**. Free on all three registries, no software of that name, says "sharp boundary" (the
quality gate), easy to say, and matches the lab's naming. CLI `halocline`, short alias to pick (avoid `hcl`, which is
HashiCorp's language). Runner-up Komatiite if the story "hotter than basalt" matters more than ease of saying it.
Reserve the PyPI and crates names once chosen; GitHub user/org names are taken for all of them, which does not matter
for a repo under `T-Crypt`.

### 11.5 Resolved 2026-11-16 (maintainer)
- **Name: Halocline.** Repo, binary and package names use `halocline`. Formal trademark search still owed before any public release.
- **Disk (§9.7) cleared.** `/mnt/models` went from 24 GB to 203 GB free (79% used): SC117 and three Qwen3.6 llama-swap entries deleted, five rarely loaded GGUFs moved to `/data` (SATA SSD). The NVMe is reserved for Halocline's artifacts and the hot llama-swap models. Put the one-off 27B BF16 reference GGUF (about 54 GB) on `/data`; it is read once. Detail in `state/changes.md` 2026-11-16.

### 11.6 Decisions closed 2026-11-16 evening (maintainer: "go with your recommendation on all of it now")
- **Repo:** `github.com/T-Crypt/halocline`, created **private**, empty. Stays private until the first release gate passes; going public is a separate maintainer decision.
- **v0 API:** OpenAI-compatible `/v1/chat/completions` and `/v1/models` with streaming, tool calls and image input; one slot (`--parallel 1`), which is how llama-swap and Berth run every model here. Grammar and strict JSON schema come later. No Anthropic dialect, no chat UI (§10.6). Status JSON and Prometheus `/metrics` land with the first serve (P4 or P5, whichever runs first).
- **KV default:** sparse runs int8 K/V at every context that fits (262K is about 3.2 GiB). Dense is chosen by the §11.1 A/B between UD-Q4_K_XL + K8V4 and UD-IQ4_XS + int8 at 262K, on whole-system KLD at depth plus the decode floor; the winner is fixed in the dense profile. bf16 K/V remains the reference mode. This replaces decision 6.
- **Downloads started:** `unsloth/Qwen3.8-27B-GGUF` @ `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` (UD-IQ4_XS, UD-Q4_K_XL) to `/mnt/models/Ollama/unsloth/Qwen3.8-27B-GGUF/`; `jesdga/basalt-kld-bases` at its current revision (27B and Flash-Next bases, `SHA256SUMS`, README) to `/data/models/halocline/kld-bases/`. Each file is checked against the Hugging Face LFS SHA-256; manifest `/data/models/halocline/manifest-p0-2026-11-16.json`.

### 11.7 P0 speed floors measured 2026-11-16
Replaces the §9.4 floor table. Evidence `evals/2026-11-16/halocline-p0-floors/README.md` (smoke and standard-v1,
one engine on the card):
- **Sparse (Strata 0.2.10 + DFlash2):** decode 86 short, 77 at 4K, 70 at 36K, 93 at 137K; prompt 1,290 at 16K, **2,400 at
  36K, 2,860 at 137K** (rerun on a quiet host; the first runs read 30 to 40% lower because of host memory pressure, which is
  itself a finding: prefill speed on this host depends on free RAM).
- **Dense (NInfer-Qwen3.8-27B):** decode 112 short, 118 at 36K, 93 at 137K; prompt 3,215 at 36K, 2,155 at 137K. Only the
  first run's prompt rates count (later runs hit NInfer's prefix cache).
- **New P5 requirement:** host RAM. Strata at 137K reached 49.5 GiB RSS and earlyoom stopped it once at 3.6% free. The
  sparse profile must fit inside the guaranteed 55 GiB engine budget (`projects/active/the workstation-memory/PLAN.md`) with a
  measured peak, not run up to the edge of the host.

## 12. Maintainer requirement 2026-11-16: Halocline ships its own toolkit, tuned for sm_89

Maintainer, verbatim: "Make sure halocline "halo" for short comes with a bench, guide, calibration, all the suspected
tools for a cuda C++ engine like strata and basalt but tuned for our sm_89 ada footprint hardware."

- **Short name `halo`.** The rename step (HANDOFF work queue) makes `halo` the CLI: one binary with subcommands
  (`halo serve`, `halo bench`, `halo calibrate`, ...), and `halocline` as the project, package and repo name.
- **Tuned for this footprint, not ported generic.** Every tool's defaults, fixtures and pass bands are set for one RTX 4090
  (sm_89, 24 GB, PCIe 4.0 x16), 64 GB host RAM inside the 55 GiB engine budget, NVMe artifacts, llama-swap and Berth as
  the front door. No multi-GPU, sm_120, Windows or DeepSeek paths in any tool (§10.4, §11.3). Strata and Basalt tools are
  read and reimplemented, never copied (§9.1), each lift logged in `LIFTS.md`.

| Tool | What it does | Starts from |
|---|---|---|
| `halo bench` | Standard protocol: frozen corpus with a SHA-256, fixed profile, decode and prompt at 4K / 36K / 137K / 262K, JSON result schema, one engine on the card under a Berth lease. Compared against the §11.7 floors and llama.cpp. Also a quick smoke mode. | `basalt-bench`, Strata `BENCHMARK_PROTOCOL.md` and `bench/prompts/standard-v1` |
| kernel micro-benches | Per-kernel time and GB/s at the shapes the models run (MMQ per IQ encoding, GDN, QSA, MoE tail, H2D copy), sm_89 only. | `native_expert_bench`, `pk_moe_bench`, `dense_window_bench`, Strata `bench/micro`, `tools/h2d_bench.cu` |
| `halo calibrate` | Measures this host (PCIe and pinned-copy rates, CPU pool rate and affinity, free VRAM after weights) and derives expert cache size, pool workers, chunk and KV settings per profile. Plan, measure, apply and exact rollback; the result is a profile file keyed by manifest, never by model name (§9.6). | `basalt-probe` (`PROBE rates`), `basalt-plan`, `auto_plan`, Strata `tools/calibrate.py` and its rollback contract |
| quality gate | KLD, PPL and top-1 against the declared BF16 reference, bands calibrated from the local BF16-vs-BF16 run (§4); on-distribution canaries for artifact comparisons; greedy identity for numerics changes. | `tools/ppl_gate.py`, `tools/greedy_identity.py`, the `jesdga/basalt-kld-bases` files |
| `halo probe` / `halo device` | GPU, driver, CUDA, clocks, power limit, VRAM, PCIe link and NVMe read rate, with a pass or fail against the sm_89 contract (§2). | `basalt-device`, `basalt-probe` |
| `halo gguf` / `halo verify` | GGUF census (tensor types and bytes per encoding) and the source lock: manifest, revision, SHA-256 and Unsloth provenance checks (§1.7). | `basalt-gguf`, `tools/gguf_reader.py`, Strata `hf_pin_fetch.py` |
| expert profiles | Build, mix and inspect the expert residency profiles the adaptive tier starts from. | `--expert-profile-save`, Strata `make_profile.py`, `make_profile_mix.py` |
| soak and serve checks | Long-run serve soak with memory and VRAM ceilings, conversation-cache parity, cancel and disconnect cases. | `tools/serve_soak.py`, Strata `tools/conversation_cache_*`, `monitor_soak.py` |
| build checks | Register spills, ISA guard, layering and include checks, and the SASS gate. | `tools/check_*`, Strata `check_cuda_spills.py`, `tools/halocline/sass_gate.py` |
| metrics page | The read-only metrics and model-stats page of §10.6, plus Prometheus `/metrics` and a Kuma monitor. | §10.6, §11.6 |
| guide | Maintainer guide: build on Arch for sm_89, install, the llama-swap entry, profiles, bench, calibrate, quality gate, troubleshooting, undo. A `guide/halocline.md` page in this repo (docs follow the work) and the repo's own `docs/`. | Basalt `docs/`, Strata `docs/` as reference only |

Placement in the phases (§5): probe, gguf/verify and the micro-benches from P1 to P3; `halo bench` and the quality gate
are the P4 and P5 exit-gate instruments; calibrate before P5's residency work; soak, metrics and the guide before P6's
migration off Strata and NInfer. A tool counts as shipped when it has a test, a `--help`, a guide section and a recorded
run on the workstation.

## 13. Maintainer mandate 2026-11-16: build it all, for this host

Maintainer, verbatim: "Build everything for my host, proof of concept, dense + sparse engine made for 1 lane of hardware,
fully optimized, leading edge papers research based, vault backed, data backed, testing and eval backed by the months
of data here in the lab + all the research done. I want it a special engine just for my device, then if it works well.
We will continue to refine it and test even more once Qwen4-27B and flash drop"

What this changes and fixes:
- **Scope is §0 to §12 as written, all of it.** One lane of hardware is the workstation's single RTX 4090 (sm_89) and its host;
  nothing in the engine or its tools targets another device (§2, §10.4, §11.3, §12).
- **Proof of concept first, then refinement.** The PoC milestone is both paths serving through llama-swap on the workstation with
  the §4 quality evidence and the §11.7 floors measured: sparse (Qwen3.8-Flash-Next) first, then dense (Qwen3.8-27B as
  the proxy), per §10.7. Qwen4-27B and its Flash sibling are a config and conversion step on top once weights exist (§1.4);
  refinement continues on them.
- **Order change toward the PoC.** The scope removals stop at what blocks a clean single-card engine: the peer tier
  (wave 2b items 1 and 2 in HANDOFF) and the program layer's multi-GPU options. Then the first Linux model load, the owed
  §10.4 greedy-identity gate against `901fd9c9` and the sparse serve come before the remaining cosmetic removals
  (`prefill_fp4`, sm_120 stubs, the router) and before the rename, so the riskiest unknown (nothing loads a model on Linux
  yet) is retired early. The rename to `halo` (§12) and the import commit follow the first working serve.
- **Evidence it draws on, by default and without being asked each time:** the research base (`research_search`, tiers and
  status cited), this repo's `evals/` (months of Strata, NInfer, llama.cpp and drafting runs), the P0 floors and
  artifact manifest (§11.7, `/data/models/halocline/`), the vault for papers and model cards, and Engram for working
  notes. A design choice that a note or a measurement can settle cites it; one that cannot is marked as an assumption to
  measure.
- **Every step stays gated** as in wave 2b: build clean, SASS or parity, CPU and GPU suites, and from P4 the bench and
  quality gate (§12). GPU work runs one engine at a time under a Berth lease; it never displaces a running NInfer or Strata
  without the lease.
