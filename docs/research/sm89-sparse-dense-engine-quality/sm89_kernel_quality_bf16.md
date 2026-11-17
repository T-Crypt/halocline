# sm_89 (Ada / RTX 4090) kernel numerical quality with BF16 reference

Research-only notes for a sparse+dense LLM inference engine targeting NVIDIA Ada (sm_89, e.g. RTX 4090 24 GB). Focus: numerical quality against a BF16 reference (KLD/perplexity/task behavior), not implementation speed except where it changes numerics. Compiled 2026-11-16.

## Scope and sources used

- Local lab records (primary evidence for NInfer sm_89 port behavior): `~/<private>/reference/records/ninfer-v3-port.md`; parity/perplexity artifacts `~/<private>/evals/2026-10-10/v3-parity/*` (inspectable on-host; summary values quoted below from those files/record).
- llama.cpp quality measurement docs and PRs: [tools/perplexity/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md), [PR #5747 IQ4_XS](https://github.com/ggml-org/llama.cpp/pull/5747), [llama.cpp wiki Feature matrix](https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix).
- FlashInfer: [arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [FlashInfer docs](https://docs.flashinfer.ai/).
- vLLM FP8 docs: [FP8 W8A8 (LLM Compressor)](https://docs.vllm.ai/en/latest/features/quantization/fp8.html).
- LMDeploy quantization docs: [AWQ/GPTQ W4A16](https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html), [repo doc](https://github.com/InternLM/lmdeploy/blob/main/docs/en/quantization/w4a16.md).
- ik_llama.cpp: [repo](https://github.com/ikawrakow/ik_llama.cpp).
- NVIDIA PTX ISA (for instruction availability; exact per-arch encoding/availability should be confirmed against the installed CUDA/PTX docs): [PTX ISA - warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
- Not found / not fully verified in this pass: ninfer prep4qwen dossiers at `~/handoffs/ninfer-4090/2026-10-13-prep4qwen-research` contain only `brief.md`, `log.tsv`, `run_target.sh` (no `docs/research/*` dossiers present on disk at time of read); no `research_*` KB notes matched "FlashInfer" or "2501.01005".

---

## Key question 1: What does "quality KLD match on BF16" mean in practice (KLD, perplexity, task accuracy) and how is it measured?

### Takeaway
In practice, "match BF16 quality" is demonstrated by small distribution/perplexity deltas versus a BF16 (or FP16) reference on the same corpus and tokenizer, often using llama.cpp's `perplexity` tool metrics (KLD of logits, PPL ratio/delta, mean/RMS/per-tail probability changes). Near-identical PPL is necessary but not sufficient; task/greedy-parity checks catch regressions PPL can hide, while KLD and per-token probability-shift tails are sensitive to quantization/kernel-induced noise.

### Cited Findings
- llama.cpp records FP16 logits to a binary file and computes Kullback-Leibler divergence (KLD) between FP16 and quantized logit distributions, where 0 means identical distributions; uncertainty is estimated assuming Gaussian behavior per token. — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- The same tool reports: ratio and difference of mean FP16 PPL vs quantized PPL (with propagated uncertainty), mean change in "correct" token probability, Pearson correlation of correct-token probabilities, percentiles of change in correct-token probability (symmetric tails suggest noise; larger negative tail suggests quality loss), RMS of change in token probabilities, and "Same top p" (how often both models assign highest probability to the same token). — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- llama.cpp documents BF16 vs FP16 comparison (LLaMA 3 8B): Mean KLD 0.00002515 ± 0.00000020, Mean PPL ratio 1.000404 ± 0.000086, RMS Δp 0.150 ± 0.001%, Same top p 99.739 ± 0.013%, illustrating that close-but-not-identical low-precision baselines still show measurable logit deltas. — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- llama.cpp perplexity guidance states perplexity is not directly comparable across different tokenizers/models and finetunes can raise PPL while human-rated quality improves; within llama.cpp PPL is mainly used to judge quality loss of quantized models vs FP16. — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- IQ4_XS PR reports quality primarily via KLD/PPL-style comparisons (e.g. KLD, PPL, mean Δp) against baselines on standard corpora, showing smaller-bpw quants traded against those metrics. — [ggml-org/llama.cpp PR #5747](https://github.com/ggml-org/llama.cpp/pull/5747)
- vLLM FP8 flow evaluates practical quality/accuracy with downstream benchmarks (example: `lm_eval` on `gsm8k`, reporting exact_match) after quantization, complementing distribution metrics with task accuracy. — [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- Local NInfer sm_89 parity evidence (Ada/4090 context per record): full-corpus perplexity 4.651043 for v3 (rk4v4-e8 path noted) vs v2 A16 4.651185 (-0.003%), with greedy outputs matching v2 up to near-ties in that stage-3 gate; this is a concrete example of BF16/A16-reference parity measurement on sm_89. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md ), [`~/<private>/evals/2026-10-10/v3-parity/ppl-v3.txt`]( ~/<private>/evals/2026-10-10/v3-parity/ppl-v3.txt )

### Inferences
- A robust "BF16-quality" claim on sm_89 should combine: (1) PPL/KLD vs BF16 on fixed corpus/tokenizer, (2) per-token tail metrics (negative Δp percentiles, RMS Δp, Same top p) to detect asymmetric degradation, and (3) greedy/task checks on representative prompts. PPL-only parity can miss localized logit damage.
- For sparse+dense engines, measure dense-only, sparse-only, and mixed routing modes separately; routing changes token distributions and can mask or expose kernel numeric issues.

### Gaps
- No single universal KLD threshold defines "match" across models/corpora in the sources gathered; acceptable bounds are empirical (compare to BF16-vs-FP16 noise floor and task sensitivity). — not found as a standard constant.
- Task-accuracy parity on sm_89 for the specific target models/engine in this project is not present in the consulted sources.

---

## Key question 2: Which sm_89 tensor-core instructions are available (mma.sync.m16n8k16 fp16/bf16, m16n8k32 fp8/int8) and how do they affect numerical quality vs BF16 reference?

### Takeaway
Ada (sm_89) exposes warp-level MMA tensor-core instructions including FP16/BF16 (`m16n8k16`-class) and FP8 (`e4m3/e5m2`, commonly `m16n8k32` forms in PTX/CUDA toolchains that target Ada) paths; INT8 MMA is also available on Ada tensor cores. Numerical quality vs BF16 depends less on the MMA mnemonic than on input format, accumulation precision, scaling/dequant, and reduction order. FP32 accumulation in FP8/low-precision MMA is typically the quality-safe choice but can be throughput-limited on Ada.

### Cited Findings
- NVIDIA PTX ISA documents warp-level matrix multiply-accumulate instructions under `mma`/`mma.sync` (and related WMMA/MMA forms), with shapes/data types selected by instruction variants and target architecture; exact availability for sm_89 must be checked against the PTX ISA for the installed CUDA version. — [NVIDIA PTX ISA: Warp Level Matrix Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
- Local NInfer port notes state: "upstream's block-scale fp8 mma was unit-scale (UE8M0 0x7f) only for Blackwell full-rate accumulation — the plain sm_89 fp8 mma (`m16n8k32 e4m3`) is numerically identical. f32 accumulation is half-rate on Ada; revisit only if the stage-6 bench says so." — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- vLLM states FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell), while Turing/Ampere run FP8 models as weight-only W8A16 via Marlin; this implies native FP8 compute paths are used on sm_89-class GPUs when selected. — [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- Hand-written PTX study on NVIDIA L4 (Ada, SM89) evaluates PTX Tensor Core GEMM kernels across FP16, INT8, and INT4 and notes the pipeline uses `cp.async`, `ldmatrix`, and `mma.sync`; it reports precision-dependent speed/behavior tradeoffs (performance-focused, but confirms `mma.sync` PTX usage on sm_89). — [arXiv:2608.10103](http://arxiv.org/abs/2608.10103v1)
- BF16 vs FP16 reference comparison in llama.cpp shows small but non-zero distribution drift (KLD 2.515e-5, RMS Δp 0.15%), indicating even "near-BF16" baselines differ measurably at logit level. — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)

### Inferences
- For BF16-reference parity on sm_89: prefer BF16 tensor-core MMA for activations/weights where possible, FP32 accumulation for reductions/epilogues, and avoid unnecessary dtype downcasts (BF16->FP16->BF16) in fused paths. When using FP8/INT8 MMA, quality is dominated by quantization/dequant/scaling and accumulation policy, not only by the MMA shape.
- On Ada, choosing FP16 accumulators for speed in FP8/FP16 MMA paths risks larger numeric drift than FP32 accumulation; treat "half-rate FP32 accum on Ada" as a performance constraint to budget, not a reason to silently lower accumulation precision in quality-critical kernels.

### Gaps
- This pass did not extract a definitive, version-pinned table from the local CUDA/PTX install showing every `mma.sync` variant (`m16n8k16` fp16/bf16, `m16n8k32` fp8/int8, accumulator types) enabled on sm_89; confirm with `ptxas`/PTX ISA for the target CUDA toolkit. — gap (tooling/version-specific).
- No direct microbenchmark numbers here for numerical error per MMA variant on sm_89 beyond the NInfer note and cited papers.

---

## Key question 3: What are known numerical pitfalls on Ada (accumulation order, f16 vs bf16, fp32 accumulation, reduction order, atomics)?

### Takeaway
The main quality risks are precision/rounding choices (FP16 vs BF16 ranges and mantissa behavior), accumulator type and reduction/tiling order (non-associative floating point), and non-deterministic or poorly controlled reductions/atomics. sm_89-specific port experience adds architecture-gated synchronization/cooperative-launch constraints that can force kernel restructures affecting numerics if not handled carefully.

### Cited Findings
- Floating-point reductions are order-dependent; changing tile traversal, split-K, warp reduction order, or atomic accumulation order can change results. (General numerical fact; implied by strict parity work and tolerance handling in kernel ports.) — inference grounded in porting practice; see NInfer tolerance/measurement note below.
- NInfer sm_89 bring-up required arch-qualifying test oracles/tolerances by measurement (e.g. gdn gating rel_l2 measured 1.434e-6 vs upstream 1.4e-6 limit, ratio 1.024 attributed to arch codegen noise; requalified to 2.0e-6 on sm_89), showing small but real arch-dependent numeric deltas. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- NInfer notes: "f32 accumulation is half-rate on Ada" for the plain sm_89 fp8 mma path discussed, creating pressure to trade throughput vs accumulation precision. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- NInfer sm_89 mechanical constraints with potential correctness/numeric side effects if mishandled: static cooperative-residency constants overflow Ada (`cudaErrorCooperativeLaunchTooLarge`) so occupancy/workspace must use runtime `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; `mbarrier.try_wait` suspend and `fence.mbarrier_init.release.cluster` are sm_90+; PDL device intrinsics must stay arch-gated. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- llama.cpp BF16 vs FP16 comparison demonstrates dtype/representation differences produce measurable KLD/PPL/probability deltas even at high precision (Mean KLD 2.515e-5, RMS Δp 0.15%). — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- llama.cpp quantization evaluation uses tail metrics (negative Δp percentiles, RMS Δp, Same top p) to distinguish "added noise" from systematic quality loss. — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- vLLM kernel selection can change backend (DeepGEMM/CUTLASS/Marlin/Triton/fallback) by hardware/quantization; backend differences can affect numerics and should be pinned/logged for parity work. — [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)

### Inferences
- Pitfalls to guard on sm_89: FP16 intermediate overflow/underflow vs BF16 dynamic range in attention/FFN epilogues; using FP16 accumulators in GEMM/attention softmax paths; non-deterministic `atomicAdd` reductions in sparse gathers/scatters or split reductions; different reduction tree shapes between prefill/decode kernels; fused epilogue reordering (bias+residual+norm) changing rounding points.
- For "BF16 reference" parity, keep reference and candidate on identical tokenization, masks, RoPE/scaling constants, attention sink/causal semantics, and deterministic reduction order where possible; report tolerances with measured arch deltas (as in NInfer) rather than copying sm_120a/oracle thresholds blindly.

### Gaps
- No comprehensive public Ada-only errata list on numeric pitfalls was collected in this pass; specific atomics/reduction nondeterminism behavior depends on kernel implementation, not only sm_89.

---

## Key question 4: How do leading engines (llama.cpp, NInfer, ik_llama.cpp, LMDeploy, vLLM, SGLang) maintain quality on sm_89?

### Takeaway
Engines maintain quality by (a) choosing BF16/FP16/FP32 accumulation policies per kernel, (b) using calibrated quantization formats and imatrix/per-group scaling, (c) validating with perplexity/KLD and task/greedy parity, and (d) selecting architecture-appropriate kernels/backends (with fallbacks on unsupported hardware). Direct, comparable sm_89 quality benchmarks across all named engines were not fully found in one place.

### Cited Findings
- **llama.cpp:** Provides `tools/perplexity` to compute PPL, KLD vs FP16 logits, PPL ratio/delta, Δp statistics, RMS Δp, and Same top p; supports importance-matrix (`imatrix`) quantization workflows to improve low-bit quality (see K-quants/I-quants comparisons in README/PR context). — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md), [PR #5747](https://github.com/ggml-org/llama.cpp/pull/5747)
- **NInfer (sm_89 port evidence):** Ported v3 runtime to sm_89 with arch-safe substitutions (plain sm_89 fp8 mma noted numerically identical in that context; cp.async/mbarrier/PDL/cooperative-launch adaptations), compiled-out nvfp4 on sm_89, and validated with op gates/ctest plus perplexity and greedy parity against v2 (reported -0.003% full-corpus PPL delta in stage-3 gate). — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md ), [v3 parity eval dir]( ~/<private>/evals/2026-10-10/v3-parity/ )
- **ik_llama.cpp:** Fork focused on quantization/performance variants (many IQ/K quant and kernel fixes in changelog); quality practice in ecosystem commonly relies on imatrix/perplexity tooling inherited from llama.cpp lineage, but this pass did not retrieve a dedicated sm_89 quality report. — [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **LMDeploy:** TurboMind documents W4A16 (AWQ/GPTQ) inference on Ada Lovelace (sm89) among supported GPUs; quantization module emphasizes AWQ and calibration/requantization options (e.g. `--search-scale`, batch size guidance) when accuracy is compromised. — [LMDeploy W4A16 docs](https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html), [GitHub doc](https://github.com/InternLM/lmdeploy/blob/main/docs/en/quantization/w4a16.md)
- **vLLM:** Supports FP8 W8A8 on compute capability >= 8.9 (Ada/Hopper/Blackwell) and W8A16 Marlin fallback on older GPUs; recommends/uses dynamic activation scaling schemes in FP8 flows and provides kernel backend selection/fallbacks (`--linear-backend`, DeepGEMM/CUTLASS/Marlin/Triton etc.) with accuracy evaluation via `lm_eval` examples. — [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- **SGLang:** Uses high-performance attention/runtime kernels and integrates with backends such as FlashInfer in the ecosystem (FlashInfer states integration with SGLang, vLLM, MLC-Engine), but this pass did not collect an sm_89-specific numeric-quality methodology document for SGLang. — [FlashInfer paper](https://arxiv.org/abs/2501.01005)

### Inferences
- Quality-preserving pattern across engines: keep a BF16/FP16 reference path for validation, pin kernel/backend choices during parity runs, and treat quantized serving paths (W4A16/W8A8/FP8 KV cache) as requiring calibration + regression checks (PPL/KLD/tasks) rather than assuming bitwise equivalence.

### Gaps
- No directly comparable, same-model/same-corpus sm_89 (RTX 4090) KLD/PPL/task table for llama.cpp vs ik_llama.cpp vs LMDeploy vs vLLM vs SGLang was found in this pass.
- SGLang-specific sm_89 kernel numerical policies (accumulation dtype, attention backend fallback rules) not verified from primary docs here.

---

## Key question 5: What dequantization strategies preserve quality (imatrix, layer-specific, per-channel scales)?

### Takeaway
Quality-preserving dequantization relies on better scale placement and calibration: importance-weighted quantization (imatrix), finer-grained scales (per-channel/per-group/blockwise), and selective precision for sensitive tensors/layers. The right granularity depends on format and kernel support; calibration data and scale search can recover accuracy when naive RTN hurts.

### Cited Findings
- llama.cpp importance matrix (imatrix) workflow improves quantization quality by weighting quantization error according to activation/importance statistics; perplexity/KLD comparisons are used to evaluate gains (e.g. `q4_K_M` with imatrix vs without in llama.cpp tables). — [llama.cpp perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md)
- llama.cpp K-quants/I-quants differ in encoding and quality/size tradeoffs; IQ formats (e.g. IQ4_XS) use non-linear/codebook-style representations in parts of the design and are evaluated with KLD/PPL/Δp metrics. — [PR #5747](https://github.com/ggml-org/llama.cpp/pull/5747)
- vLLM FP8 dynamic scheme description: static per-channel quantization on weights and dynamic per-token quantization on activations (FP8_DYNAMIC/RTN flow in docs), avoiding calibration data in that specific simple flow while preserving accuracy for supported cases. — [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- LMDeploy W4A16 docs recommend enabling `--search-scale` for re-quantization and increasing batch size when quantized model accuracy is compromised, indicating scale-search/calibration sensitivity in W4A16 quality. — [LMDeploy W4A16 docs](https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html)
- FlashInfer targets efficient attention/KV-cache handling across formats; quantized KV/attention paths in modern serving stacks often combine FP8/BF16 compute with careful scaling (specific sm_89 dequant recipes not detailed in the abstract scraped). — [FlashInfer arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [FlashInfer docs](https://docs.flashinfer.ai/)

### Inferences
- For sm_89 BF16-reference parity: keep embeddings, output/lm_head, norms, and attention-critical projections at higher precision or finer quantization if needed; use per-group/per-channel scales over per-tensor when kernels allow; prefer imatrix-calibrated GGUF/weight-only quantization for CPU/GPU mixed deployments where supported.
- Dequant should happen as close to MMA/compute as possible in the same precision domain expected by the kernel (avoid extra casts), and scales must be applied consistently with the MMA/accumulation path (unit-scale assumptions differ across architectures, as seen in NInfer's Blackwell-vs-Ada note).

### Gaps
- No single source here quantifies layer-wise sensitivity for the target sparse+dense model family on sm_89.
- Per-channel vs per-group vs blockwise quality deltas for LMDeploy/vLLM/SGLang on Ada were not measured in this pass.

---

## Key question 6: What are the sm_89-specific kernel patterns that avoid quality loss (cp.async vs TMA, standard barriers vs cluster barriers)?

### Takeaway
On sm_89, use Ada-supported async-copy and synchronization primitives (`cp.async` pipelines, standard `mbarrier`/barrier usage valid for sm_89) and avoid Hopper/Blackwell-only features (TMA, cluster-scope barrier forms, PDL intrinsics) unless explicitly guarded. Correct architecture gating prevents illegal instruction paths and forced rewrites that can accidentally change reduction order, tiling, or accumulation behavior.

### Cited Findings
- NInfer sm_89 port replaced/ported TMA-based paths with `cp.async` for bf16 and fp8 a8 kernels ("2× TMA→cp.async ports"), and used sm_89-safe mbarriers with PDL no-ops on Ada. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- NInfer notes: `mbarrier.try_wait` (suspend form) and `fence.mbarrier_init.release.cluster` are sm_90+; shared helpers are arch-gated. PDL device intrinsics must remain arch-gated because upstream kernels call them but fork/sm_89 paths must not. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- NInfer required runtime cooperative occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) because static cooperative-residency constants overflow Ada, and workspace-capacity functions must read the same value to avoid contract/test drift. — [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md )
- PTX `cp.async` and warp-level MMA (`mma.sync`/`ldmatrix`) are part of the low-level tensor-core pipeline used in high-performance kernels on Ada-class GPUs (as described in sm_89 PTX GEMM study context). — [arXiv:2608.10103](http://arxiv.org/abs/2608.10103v1), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### Inferences
- Quality-preserving sm_89 pattern: keep mathematically equivalent tiling/accumulation order when substituting TMA->`cp.async`; gate all sm_90+/cluster/TMA/PDL code paths at compile/runtime; validate that occupancy/capacity-driven launch changes do not alter split/reduction structure. Prefer deterministic epilogue reductions and avoid architecture-specific fast paths that change numerics unless separately qualified.
- Sparse kernels should avoid atomic accumulation into shared output buffers when determinism/bitwise-stable parity matters; if atomics are required, document ordering and validate with repeated-run parity/KLD.

### Gaps
- No direct microarchitectural guidance here on `cp.async` vs async proxy behavior affecting numerical rounding (they affect memory movement/scheduling; numerics usually preserved if math/order unchanged), but empirical parity evidence for a specific kernel is required.

---

## Key question 7: Papers/repos on HuggingFace or Chinese sources relevant to sm_89 quality

### Takeaway
Relevant public sources exist, but sm_89-specific numerical-quality studies are sparse compared to performance studies. The most actionable sources found are llama.cpp/ik_llama.cpp quality tooling, vLLM/LMDeploy quantization docs with Ada support statements, FlashInfer serving architecture, and Ada (SM89) PTX/kernel papers. Chinese-engine primary quality docs (e.g. FastLLM) were not concretely retrieved in this pass.

### Cited Findings
- FlashInfer (serving attention engine, integrated with SGLang/vLLM/MLC-Engine): [arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [docs](https://docs.flashinfer.ai/), [repo](https://github.com/flashinfer-ai/flashinfer).
- llama.cpp quantization quality measurement and imatrix/KLD practice: [perplexity README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md), [IQ4_XS PR #5747](https://github.com/ggml-org/llama.cpp/pull/5747).
- ik_llama.cpp (quant/kernel fork with active fixes): [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).
- vLLM FP8 on Ada (sm89/sm_89 compute capability >= 8.9): [FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html).
- LMDeploy W4A16 on Ada Lovelace (sm89): [W4A16 docs](https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html), [GitHub doc](https://github.com/InternLM/lmdeploy/blob/main/docs/en/quantization/w4a16.md).
- Ada/SM89 PTX tensor-core kernel study (performance + low-level pipeline): [Hand-Written PTX Tensor-Core GEMM Kernels on NVIDIA L4 (Ada, SM89)](http://arxiv.org/abs/2608.10103v1).
- Local sm_89 parity/port record (NInfer): [`~/<private>/reference/records/ninfer-v3-port.md`]( ~/<private>/reference/records/ninfer-v3-port.md ).

### Inferences
- For Chinese-source follow-up, prioritize LMDeploy (InternLM) primary docs/issues/commits for sm89 W4A16/AWQ behavior, and search FastLLM repos/docs/issues for Ada/4090 FP8/BF16/perplexity reports; treat community claims as T4 until reproduced with KLD/PPL on a fixed corpus.

### Gaps
- FastLLM-specific sm_89 quality/KLD evidence: not found in this pass.
- HuggingFace-hosted papers explicitly benchmarking "BF16 KLD match on sm_89" for sparse+dense inference: not found in this pass (searches returned adjacent topics).
- ninfer prep4qwen dossiers for vLLM/SGLang under `~/handoffs/ninfer-4090/2026-10-13-prep4qwen-research`: not present as completed dossier files at read time (only brief/log/run script found).

---

## Cross-cutting recommendations for a BF16-reference sm_89 sparse+dense engine

- **Reference discipline:** keep a BF16 reference path/kernel config and compare on identical tokenizer, prompts/corpus, sampling/greedy settings, and KV/attention semantics.
- **Metrics:** report PPL, KLD vs reference logits, PPL ratio/delta, mean/RMS Δp, negative-tail Δp, Same top p, and at least one task/greedy parity suite.
- **Accumulation:** prefer FP32 accumulation in GEMM/attention reductions; if using FP16/fast accumulators on Ada, gate behind explicit quality opt-in and re-measure.
- **Architecture gating:** forbid sm_90+/Blackwell-only features on sm_89 (TMA/cluster barrier forms/PDL paths) and validate substituted paths (`cp.async`, standard barriers) preserve reduction/math order.
- **Quant/dequant:** use imatrix or calibrated scales, finer-grain scales (per-channel/per-group/block) where available, selective high precision for sensitive layers, and consistent scale application matching the MMA path.
- **Determinism:** control reduction order and minimize nondeterministic atomics in parity mode; log selected kernel/backend.

## Evidence quality note

- Local NInfer parity numbers are from on-host eval artifacts/record and should be treated as project-local measurements (T0-like in lab context) but re-verifiable only where artifacts remain.
- Web/docs findings are cited to primary-ish sources (tool docs, arXiv, vendor docs/repos) but were not independently reproduced in this run.
- "Not found" items are called out in each Gaps subsection; no missing metrics were invented.