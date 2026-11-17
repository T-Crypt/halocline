# Dense (and Dense-Hybrid GDN) Execution on sm_89 (Ada) with BF16-Quality Focus

## What dense architectures are relevant (Qwen3.8-27B dense: 64 layers, 48 GDN + 16 full attention, GQA, MTP)?

### Takeaway
The primary reference architecture for this request is the Qwen3.5/3.8-family 27B dense hybrid: 64 transformer blocks arranged as 16 repeats of 3 Gated DeltaNet (linear-attention) layers followed by 1 full gated-attention layer (48 GDN + 16 full attention), with GQA in full-attention layers, partial/MRoPE settings in Qwen3.5 text config, and a small MTP head in the published configs/artifacts. On sm_89, quality-preserving execution means matching this layout, state semantics, and BF16-capable numerics rather than treating it as a standard dense Transformer.

### Cited Findings
- Qwen3.8-27B is described on Hugging Face as a dense hybrid built on the Qwen3.5 architectural foundation with hidden layout `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`; GDN: 48 V heads / 16 QK heads, head dim 128; gated attention: 24 Q heads / 4 KV heads, head dim 256, RoPE dimension 64. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Qwen3.5-27B raw config (`text_config`) shows `num_hidden_layers: 64`, `layer_types` with a repeating `[linear_attention, linear_attention, linear_attention, full_attention]` pattern (48 linear + 16 full), `full_attention_interval: 4`, `linear_num_value_heads: 48`, `linear_num_key_heads: 16`, `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `linear_conv_kernel_dim: 4`, `num_attention_heads: 24`, `num_key_value_heads: 4`, `head_dim: 256`, `attn_output_gate: true`, `mamba_ssm_dtype: "float32"`, `mtp_num_hidden_layers: 1`, `max_position_embeddings: 262144`, and `rope_parameters` including `partial_rotary_factor: 0.25`, `rope_theta: 10000000`, `mrope_interleaved: true`, `mrope_section: [11,11,10]`. — [Qwen/Qwen3.5-27B config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)
- Qwen3.5 documentation states: “It uses a 3:1 hybrid attention stack — three Gated DeltaNet (linear attention) layers for every one Gated Attention (full attention) layer…” and that the page covers dense Qwen3.5 variants including Qwen3.5-27B. — [Qwen3.5 model docs (Transformers)](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)
- NInfer-4090/RTX 4090 ports and related artifacts explicitly target Qwen3.8-27B on one 24 GB sm_89 GPU, loading groupwise `.ninfer` artifacts and supporting MTP, paged KV, ReplaySSM state transactions, vision, etc. — [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows), [pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer](https://huggingface.co/pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer)
- NVIDIA’s ReplaySSM post names Qwen3.5 through Qwen3.8 among hybrid models using Mamba-2, Gated DeltaNet (GDN), or Kimi Delta Attention as the SSM layer. — [ReplaySSM: Rethinking State Storage for Faster Hybrid-Model Decoding](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/)

### Inferences
- “Qwen3.8-27B dense” in engine/runtime contexts (especially NInfer ecosystem) should be treated as the same class of 3:1 dense GDN/full-attention hybrid as Qwen3.5-27B unless a specific config proves otherwise; the HF Qwen3.8-27B card exposes the same 48/16 split and head dimensions as Qwen3.5-27B docs/config snippets. BF16-quality dense execution on sm_89 must preserve the 64-layer schedule, GQA ratios, RoPE/position settings, and MTP semantics where enabled.
- Dense-hybrid here means “dense FFN weights, hybrid attention mechanism” (not MoE), which simplifies expert-routing concerns but keeps recurrent-state quality, KV behavior, and mixed-layer scheduling as first-order correctness issues.

### Gaps
- I did not retrieve a canonical raw `config.json` for `Qwen/Qwen3.8-27B` from Hugging Face in this run (the model card text was available via search snippets but direct raw config fetch was not completed), so exact parity/differences versus Qwen3.5-27B (e.g. tokenizer IDs, rope details, MTP fields) cannot be exhaustively confirmed from primary config files for both models.
- No authoritative “Qwen3.8 tech report” PDF/HTML with full architecture ablations was found beyond HF card/blog references in available sources.

## How does GDN (Gated Delta Net) recurrence affect quality on sm_89 (fp32 state, conv kernel, output gate)?

### Takeaway
GDN quality is dominated by recurrent-state fidelity: decay (`alpha`), delta/update (`beta`), key/query normalization, causal depthwise convolution mixing, and output gating all interact over long horizons. On sm_89, preserving BF16 model weights is not sufficient if recurrent state, conv intermediates, or gate math are aggressively quantized or accumulated poorly; Qwen3.5’s config explicitly indicates `float32` SSM dtype and a short conv kernel, which are strong hints for quality-sensitive implementation choices.

### Cited Findings
- Qwen3.5-27B text config specifies `mamba_ssm_dtype: "float32"`, `linear_conv_kernel_dim: 4`, `attn_output_gate: true`, and GDN head geometry (`linear_num_key_heads: 16`, `linear_num_value_heads: 48`, head dims 128). — [Qwen/Qwen3.5-27B config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)
- The Gated DeltaNet formulation combines a gated decay with a delta-rule update over a matrix state `S_t`, using per-head gates (`alpha`, `beta`) and producing outputs via query projection of the recurrent state; GDN is described as a recurrent linear attention mechanism keeping fixed-size running state per request. — [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v1), [What Is Gated DeltaNet (GDN)?](https://inferencex.semianalysis.com/glossary/gated-deltanet)
- A reference implementation/analysis shows GDN components including depthwise causal Conv1D + SiLU on mixed Q/K/V, L2 normalization of Q/K, sigmoid `beta`, exponential decay `alpha` derived from `A_log/W_alpha/dt_bias`, recurrent state update `S = S * a_t + ...`, gated RMSNorm/output gating (`rms_norm(output) * silu(z)` / gate) and output projection. — [Gated DeltaNet for Linear Attention (LLMs-from-scratch)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md), [Qwen3.5 Gated DeltaNet Linear Attention — Technical Analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- NVIDIA ReplaySSM explains hybrid-model SSM decoding state traffic/update behavior and represents SSM state via checkpoint plus recent-input buffer/replay (reducing state memory traffic and simplifying rollback/speculation concerns), relevant to correctness/perf tradeoffs in SSM-heavy hybrids like Qwen3.5–3.8. — [ReplaySSM: Rethinking State Storage for Faster Hybrid-Model Decoding](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/), [ReplaySSM blog (Tri Dao)](https://tridao.me/blog/2026/replayssm/)
- NInfer-4090 ports advertise “ReplaySSM state transactions” as part of the Qwen3.8 runtime on sm_89, indicating engine-level handling of recurrent SSM state/rollback semantics. — [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)

### Inferences
- On sm_89, BF16-quality GDN execution should keep recurrent state (`S`) and gate-sensitive accumulations in FP32 (or otherwise proven-equivalent precision), especially for long contexts/speculative rollback paths; this aligns with `mamba_ssm_dtype=float32` in Qwen3.5 config and the recurrent nature of GDN updates.
- The depthwise causal conv (`kernel_dim=4`) and output gate are not cosmetic: conv ordering/padding/activation and `norm * silu(gate)` fusion details can perturb the state trajectory token-by-token. For quality, prefer reference-matched kernels/fusions and avoid ad hoc approximations (e.g. altered norm epsilon behavior, gate dtype truncation, or conv fusion that changes numerics).
- Replay-style/checkpointed state management can affect both throughput and correctness under speculative decoding or chunked serving; on Ada engines without Hopper-specific GDN kernels, state transaction correctness is as important as raw kernel speed.

### Gaps
- No public, sm_89-specific ablation quantifying quality loss from FP16/BF16 GDN state vs FP32 state for Qwen3.5/3.8-27B was found.
- Detailed NInfer “replayssm” implementation internals (exact checkpoint format, rollback granularity, precision of stored intermediates) were not available in fetched primary docs beyond README-level claims.

## What attention patterns are needed (GQA, partial RoPE, MRoPE) and how to preserve quality on sm_89?

### Takeaway
The full-attention layers in this hybrid use grouped-query attention and nontrivial positional encoding settings (including partial rotary factor and MRoPE-related fields in Qwen3.5 config). Quality preservation on sm_89 requires exact mask/layout semantics, correct RoPE scaling/application (including partial/MRoPE sectioning/interleaving), and KV-cache precision/layout choices that do not silently degrade long-context behavior.

### Cited Findings
- Qwen3.5-27B config (full-attention): `num_attention_heads: 24`, `num_key_value_heads: 4` (GQA), `head_dim: 256`, `attn_output_gate: true`; positional config includes `rope_parameters.partial_rotary_factor: 0.25`, `rope_theta: 10000000`, `mrope_interleaved: true`, `mrope_section: [11,11,10]`, and model `max_position_embeddings: 262144`. — [Qwen/Qwen3.5-27B config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)
- Qwen3.8-27B HF card states gated attention with 24 Q / 4 KV heads, head dim 256, rotary position embedding dimension 64. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- vLLM attention backend documentation (CUDA) shows backend/compute-capability-dependent selection and feature constraints (e.g. FA2/FA3/FA4, Triton/FlashInfer paths, KV dtype/block-size/head-size support), implying engine/backend choice on Ada can change kernel path and numerics/feature availability. — [Attention Backend Feature Support (vLLM)](https://docs.vllm.ai/en/latest/design/attention_backends/)
- SGLang documents automatic/explicit attention backend selection and hybrid GDN behavior where full-attention layers still require a standard `--attention-backend`; platform constraints differ by architecture (and GDN kernel selection is separate via `--linear-attn-backend`). — [Attention Backend (SGLang)](https://docs.sglang.io/docs/advanced_features/attention_backend)

### Inferences
- For BF16-quality on sm_89, implement/validate GQA expansion semantics, attention output gating (if present in runtime path), and exact RoPE application (partial rotary factor and MRoPE interleaving/sectioning) against reference outputs; errors here often appear as long-context or position-sensitive regressions rather than obvious crashes.
- KV-cache dtype/layout is a quality knob: INT8/FP8/E8 or aggressive block quantization can extend context on 24 GB cards (as seen in NInfer ecosystem claims) but must be validated against BF16/reference behavior, especially for needle retrieval and long reasoning chains.
- Because only 16/64 layers are full attention in this architecture, attention-kernel quality issues may be masked in short prompts but dominate retrieval/localization tasks that rely on those layers.

### Gaps
- I did not find an authoritative, sm_89-specific kernel recipe (exact FA/Triton kernel + KV dtype/layout) proven to be numerically equivalent to BF16 reference for Qwen3.5/3.8-27B full-attention layers.
- No source in this run explicitly documented how MRoPE section/interleaving is handled in NInfer’s custom runtime path for Qwen3.8-27B multimodal/text contexts.

## How do leading engines run dense hybrids on sm_89 (NInfer replayssm, ik_llama.cpp, vLLM/SGLang Triton fallback)?

### Takeaway
On Ada/sm_89, mature engines generally avoid Hopper/Blackwell-only specialized GDN kernels and instead use custom sm_89 implementations (NInfer), ggml/CUDA fused delta-net paths (ik_llama.cpp), or Triton/FLA-based linear-attention backends (SGLang/vLLM ecosystem), with FlashInfer’s GDN support currently centered on newer architectures. Quality and capability therefore depend on which backend path is actually selected.

### Cited Findings
- **SGLang:** GDN is automatically activated for supported hybrid models; GDN linear-attention kernels are selected via `--linear-attn-backend` (default Triton). SGLang docs list FlashInfer GDN as “CUDA, SM90/SM100/SM103” for decode/prefill (with conditions), while Triton supports CUDA decode/prefill broadly (including Ada-class CUDA in general Triton path), and full-attention layers still use a separate `--attention-backend`. — [Attention Backend (SGLang)](https://docs.sglang.io/docs/advanced_features/attention_backend)
- **FlashInfer:** issue/roadmap tracking shows GDN work focused on Hopper/Blackwell paths (e.g. prefill/decode kernels and SM90+ issues/discussions); issue #1690 summarizes Hopper prefill done and Blackwell/Hopper+Blackwell decode/prefill work tracked around SM90/Blackwell contexts. — [[Qwen3-Next] Gated Delta Net #1690](https://github.com/flashinfer-ai/flashinfer/issues/1690)
- **vLLM:** vLLM blog states support for Qwen3-Next hybrid attention by integrating Triton kernels from Flash Linear Attention (FLA) and using a hybrid KV cache manager for linear and full attention layers; backend docs show architecture/capability-dependent selection and Triton/FlashInfer/FlashAttention options for attention (GDN-specific path is not detailed on the fetched backend feature page). — [vLLM Now Supports Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next), [Attention Backend Feature Support (vLLM)](https://docs.vllm.ai/en/latest/design/attention_backends/)
- **ik_llama.cpp:** merged fused delta-net work for Qwen3-Next and Qwen3.5-MoE (PR #1315, #1333) improved performance and later became default-on in the PR narrative; README lists “Fused delta-net for Qwen3-Next and Qwen3.5-MoE” among features. — [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [PR #1315](https://github.com/ikawrakow/ik_llama.cpp/pull/1315), [PR #1333](https://github.com/ikawrakow/ik_llama.cpp/pull/1333)
- **NInfer (sm_89 ports):** NInfer-4090 targets Qwen3.8-27B on one RTX 4090 (`sm_89`), uses groupwise `.ninfer` artifacts (groupwise-int path on Ada, not Blackwell NVFP4/W4A4), and advertises ReplaySSM state transactions, MTP, paged KV, and (in forks) E8/rotated KV modes and other sm_89-specific optimizations. — [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows), [pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer](https://huggingface.co/pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer)

### Inferences
- For dense-hybrid GDN on sm_89 today, Triton/FLA-style or custom fused kernels are the practical quality-preserving baseline when FlashInfer GDN kernels are unavailable for Ada; engine parity must be judged by numerics (state/conv/gating/KV) rather than API compatibility alone.
- NInfer’s specialized `.ninfer` pipeline can trade portability for sm_89-specific memory/layout optimizations (including KV/state handling), so BF16-quality claims should be validated against the original HF/BF16 reference model behavior, not assumed from artifact format.
- ik_llama.cpp’s fused delta-net primarily targets performance but changes the execution path for GDN layers; any BF16-quality assessment on Ada should pin commit/version and compare against unfused/reference behavior where possible.

### Gaps
- No direct, up-to-date vLLM documentation page was fetched in this run that explicitly states “GDN on sm_89 uses Triton/FLA fallback” as a normative backend rule; inference is based on vLLM’s stated FLA/Triton integration and absence of Ada-targeted FlashInfer GDN evidence.
- Public, reproducible quality comparisons (perplexity/task/NIAH) across SGLang/vLLM/ik_llama/NInfer on the same Qwen3.5/3.8-27B checkpoint on sm_89 were not found.

## What are the quality risks in dense execution (accumulation, state precision, conv, gating)?

### Takeaway
The dominant quality risks for dense GDN hybrids on sm_89 are recurrent-state precision drift, numerically sensitive conv/gating paths, accumulation/order effects in fused kernels, and KV/cache quantization interactions in the 16 full-attention layers. Because errors compound across recurrent steps and mixed layer types, small kernel-level deviations can produce large long-context or reasoning-quality regressions.

### Cited Findings
- Qwen3.5-27B config indicates `mamba_ssm_dtype: "float32"` and `attn_output_gate: true`, highlighting state and gating as explicitly modeled components in the reference architecture. — [Qwen/Qwen3.5-27B config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)
- GDN recurrence updates a persistent state with decay and delta-rule correction (`S_t`), and implementations include depthwise causal Conv1D, L2 norm on Q/K, beta/alpha gates, and gated RMSNorm/output gating (`norm * silu(gate)`) before output projection. — [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v1), [Gated DeltaNet for Linear Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md), [Qwen3.5 Gated DeltaNet Linear Attention — Technical Analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- FlashInfer GDN-related issue discussions show real-world correctness/perf pitfalls in kernel paths (e.g. hangs/architecture constraints in certain contexts), underscoring that kernel/backend selection can affect correctness, not just speed. — [[Bug] [vllm] Hang in chunk_gated_delta_rule... #3329](https://github.com/flashinfer-ai/flashinfer/issues/3329), [[Qwen3-Next] Gated Delta Net #1690](https://github.com/flashinfer-ai/flashinfer/issues/1690)
- Quantization-focused analysis for Qwen3.5 reports that quantizing linear-attention components can be more problematic for longer generations/sequences in some settings, while short-sequence metrics may look similar. — [Qwen3.5 Quantization: Similar Accuracy, More Thinking](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy), [Qwen3.5 GGUF Benchmarks (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- NInfer ecosystem materials show aggressive KV quantization modes (e.g. int8/E8/rotated variants) to fit long contexts on 24 GB Ada, which can alter memory/quality tradeoffs relative to BF16 KV baselines. — [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer](https://huggingface.co/pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer)

### Inferences
- **State precision:** store/update GDN recurrent state in FP32; avoid BF16/FP16 state checkpoints unless empirically validated. Accumulation order in fused kernels (matmul + gate application + norm) should match reference semantics as closely as possible.
- **Conv kernel:** depthwise causal conv (`k=4`) boundary handling, padding/cache warmup, and SiLU placement must be exact; off-by-one/history bugs manifest as subtle context corruption.
- **Gating:** `alpha/beta` parameterization, clamping/exp/softplus/sigmoid order, and output gate fusion (`rms_norm * silu(gate)`) are numerically sensitive; keep these in higher precision and verify against reference layer outputs.
- **Attention/KV side (16 layers):** GQA, partial RoPE/MRoPE, and KV quantization/layout can create quality cliffs at long context even when short-context perplexity is stable.
- **Speculative decoding/MTP:** hybrid state rollback and draft/verify interactions (and engine-specific ReplaySSM behavior) add extra correctness surface; “same weights” does not imply same decode distribution.

### Gaps
- No sm_89-specific, per-op error budget (state dtype, conv dtype, gate dtype, accumulation dtype) for Qwen3.5/3.8-27B was found in primary sources.
- Evidence is limited on how NInfer’s custom quantized KV/state formats compare to BF16-reference quality on standardized long-context tasks.

## What validation is needed (perplexity, task accuracy, NIAH, greedy vs sampling agreement)?

### Takeaway
BF16-quality validation for dense GDN hybrids on sm_89 must combine short- and long-context metrics: perplexity/logprob parity, deterministic greedy agreement, sampling-distribution checks, task accuracy, and retrieval/long-context stress tests (especially NIAH-style). Hybrid recurrence means passing only short perplexity or a few benchmarks is insufficient.

### Cited Findings
- Gated DeltaNet paper and follow-on discussions emphasize that linear/recurrent attention variants can differ materially on in-context retrieval/long-context behavior (e.g. NIAH/S-NIAH analyses in GDN literature), so retrieval-sensitive evaluation is relevant beyond standard LM perplexity. — [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v1)
- Qwen3.5 quantization/community benchmark reports show that long-generation behavior and some workloads can diverge from short-context/perplexity signals; Unsloth notes perplexity/KLD can be misleading depending on calibration/eval setup. — [Qwen3.5 Quantization: Similar Accuracy, More Thinking](https://kaitchup.substack.com/p/qwen35-quantization-similar-accuracy), [Qwen3.5 GGUF Benchmarks (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- vLLM/SGLang backend matrices expose multiple kernel/backend choices by architecture/capability and feature set; backend changes can alter numerics and supported features, motivating regression validation after backend or kernel-path changes. — [Attention Backend Feature Support (vLLM)](https://docs.vllm.ai/en/latest/design/attention_backends/), [Attention Backend (SGLang)](https://docs.sglang.io/docs/advanced_features/attention_backend)
- NInfer sm_89 ports report measured throughput/context results and (in forks/windows ports) some quality checks such as perplexity comparisons and task evals for specific artifacts/builds, illustrating the kind of artifact-specific validation practitioners already run. — [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows)

### Inferences
- **Minimum BF16-reference parity suite (recommended):** (1) tokenizer-aligned logprob/perplexity on fixed corpora at multiple lengths; (2) greedy decode exact-match/token-match rate on deterministic prompts; (3) sampling agreement (token distribution KL/JS, top-k overlap, and end-task variance) under fixed seeds/params; (4) task accuracy on representative coding/reasoning/instruction sets; (5) long-context retrieval/NIAH and multi-needle variants across depths/lengths up to target context.
- **Hybrid-specific checks:** compare outputs with/without speculation (MTP/draft), with/without CUDA graphs, and across chunk sizes/prefix-cache states to catch recurrent-state/rollback bugs. Validate GDN-heavy long generations separately from short prompts.
- **sm_89-specific checks:** validate selected backend path (e.g. Triton vs FlashInfer-unavailable fallback, custom kernels) because “same model” can run different code on Ada than Hopper/Blackwell.

### Gaps
- No standardized, community-accepted “BF16-quality acceptance threshold” for Qwen3.5/3.8-27B on sm_89 (e.g. max perplexity delta, min greedy match rate, required NIAH pass rates) was found.
- Public evidence is sparse for MTP/speculative-decoding parity tests specific to dense GDN hybrids on Ada.

## Papers/repos on HuggingFace or Chinese sources for dense hybrid quality on Ada

### Takeaway
Useful primary sources cluster into: (1) Qwen model/config/docs on Hugging Face, (2) GDN/linear-attention papers and kernel libraries, (3) sm_89-focused NInfer repos/artifacts (including Chinese/community forks), and (4) engine backend docs/issues clarifying Ada fallback behavior. Direct “Ada quality benchmark” papers are scarce; most Ada-specific evidence is engineering reports/repos rather than formal evaluations.

### Cited Findings
- **Hugging Face / model sources:** Qwen3.5-27B and Qwen3.8-27B model cards/config/docs; Qwen3.8-27B NInfer-converted artifact (`pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer`) targeting `UDPSendToFailed/ninfer-4090` on 24 GB RTX 4090 (`sm_89`). — [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [Qwen/Qwen3.5-27B config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json), [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3.5 docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer](https://huggingface.co/pyros-vault/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NInfer)
- **Papers / technical foundations:** Gated Delta Networks paper; Qwen3.5 blog; NVIDIA ReplaySSM blog/post; FlashQLA blog (Qwen/GDN prefill kernels context). — [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v1), [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5), [ReplaySSM (NVIDIA)](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/), [FlashQLA](https://qwen.ai/blog?id=flashqla)
- **Repos / kernels / engines:** FLA (`fla-org/flash-linear-attention`), FlashInfer (GDN issues/roadmap), SGLang and vLLM docs, ik_llama.cpp fused delta-net PRs, NInfer sm_89 forks (`sergiuszm/ninfer-4090`, `JGamboa/ninfer-4090-windows`, upstream lineage references). — [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [flashinfer-ai/flashinfer #1690](https://github.com/flashinfer-ai/flashinfer/issues/1690), [SGLang attention backend](https://docs.sglang.io/docs/advanced_features/attention_backend), [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/), [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090), [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows)
- **Chinese/community-adjacent sources surfaced in search:** Qwen3.8/Qwen3.5 ecosystem posts/articles and Chinese-origin/community model derivatives and discussions around Qwen3.8-27B/NInfer usage appeared in search results (e.g. ModelScope listing for NInfer ternary artifact, Reddit/Zhihu-style mirrors not exhaustively captured as primary docs). — [yudssd/ninfer-ampere-ternary (ModelScope search hit)](https://www.modelscope.cn/models/yudssd/ninfer-ampere-ternary), [Qwen3.8-27B on 16GB VRAM... (Reddit)](https://www.reddit.com/r/LocalLLM/comments/1wir8ky/qwen3827b_on_16gb_vram_upcoming_32gb_ram/)

### Inferences
- For sm_89 quality-focused work, combine official model/config sources (HF + Qwen docs) with engine-specific implementation evidence (SGLang/vLLM backend docs, NInfer/ik_llama repos) because Ada behavior is often determined by fallback/custom kernel paths rather than the model card alone.
- Chinese/community NInfer artifacts and forks are operationally relevant for Ada deployment details, but their quality claims should be treated as artifact-specific until reproduced against BF16 baselines.

### Gaps
- I did not find a formal paper dedicated to “dense GDN hybrid quality on Ada/sm_89” with controlled BF16-reference evaluations; most Ada-specific material is repo/readme/blog/report level.
- I did not exhaustively collect Chinese-language primary technical writeups (e.g. Zhihu/Bilibili/blog posts) beyond search snippets due source/tooling/time constraints; a targeted Chinese-source pass could improve coverage of local Ada tuning practices.
