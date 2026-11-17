# Qwen4, sparse use cases, and the dense and sparse switch on sm_89

**Date:** 2026-11-16
**Scope:** one NVIDIA Ada GPU, sm_89, RTX 4090 24 GB proving ground. Nothing here assumes sm_86, sm_90, sm_100, sm_120 or a multi-arch build.
**Role:** detail behind `BUILD_SPEC.md` §1.4, §1.5 and §1.6. Where this note and `BUILD_SPEC.md` differ, `BUILD_SPEC.md` wins.

Evidence labels follow the research base convention. T3 means an official artifact or teardown, T4 means a community lead and is not evidence. `draft` means written by a model and not reviewed. Claims from the research base are labelled at the point of use.

## 1. Qwen4, what is actually verified on 2026-11-16

The primary vendor source is the Qwen4-Max blog (`https://qwen.ai/blog?id=qwen4-max`, page dated 2026/11/15, general availability reported 2026-11-16). It states:

- Qwen4-Max is built on the new Qwen4 architecture: 2.4T backbone parameters, 71B activated per token, 328B additional n-gram embedding parameters, native vision, 1M token context.
- The architecture is the same three changes Flash-Next previewed: GDN plus QSA in the attention stack, four-branch gated residual, and n-gram embedding at layer 2. Across knowledge, code, QA and STEM, the base model reports 5 to 13 percent lower bits per byte than Qwen3.8-Max-Base while activating 25 percent fewer parameters.
- It is API only on QwenCloud. No weights. The blog closes with "Smaller models in the family are coming to open source", with no size, date or license.

Independent reporting on 2026-11-16 agrees: no Qwen4 repository in the Qwen Hugging Face org, no published weights for any Qwen4 model, and the smaller open-weight release promised "soon" without detail. Qwen4-27B comes from keynote slides and from optimization code in hardware repos. Writeups describe a dense 27B with hidden 5120, four-token MTP and QSA tuned for 24 GB, but those come from kernel strides and community posts, not a model card or config. Treat that as a T4 lead. It is a useful hint about where Qwen is going and not a spec input.

Consequence for the engine stays the same as `BUILD_SPEC.md` §1.4: the verified `qwen4_exp` config is the shape spec for the family until open weights land. Do not bake blog parameter counts into code.

Research base notes used here, all T3 with status `draft`:

- `doc:prep4qwen/qwen38-flash-next`: architecture and serving dossier, including the llama.cpp accuracy table and the consumer hardware runs.
- `doc:prep4qwen/qwen-official`: Qwen's public architecture signals.
- `doc:prep4qwen/targets`: engine target queue and sm_89 support status.

Each number taken from those notes is marked below. They are not measurements we made.

## 2. What each sparse mechanism is for, and where it earns its keep on sm_89

The architecture is three ideas layered together, not one "sparse mode". Each idea has a different best use case and a different failure mode.

| Component | What it does | Best use case | Consequence on one sm_89 24 GB card | Quality invariant |
|---|---|---|---|---|
| Gated DeltaNet, 3 of every 4 layers | Compresses history into a fixed size recurrent state | Long prefixes and streams where a growing KV cache is the limit | Removes KV VRAM growth for 36 of 48 layers | FP32 state, exact conv k=4 boundaries, alpha/beta math, gate fusion, ReplaySSM rollback |
| QSA sparse attention, the fourth layer | Scores context at micro-block granularity with a compressed indexer, then attends to a fixed 512 blocks or 2048 tokens | Retrieval and reasoning at 64K and beyond, where dense attention cost and KV reads dominate | Bounded attention and KV reads are what make 262K to 1M plausible on 24 GB; the indexer itself is small | top-512 over 4-token blocks, r=4 AvgPool before RoPE, mandatory tail block, per-layer indexer |
| MoE, 512 experts, 10 routed plus 1 shared | Keeps capacity in total parameters while activating a small slice per token | Capacity bound work when inactive experts can live in RAM or on disk and latency is not the tight constraint | Flash-Next does not fit in 24 GB at any published precision, so on one 4090 MoE means tiering and a latency cost | router logits, top-10+1 order, aggregation order and dtype identical across all tiers |
| PLE n-gram table, 51B | Adds a large local context lookup at layer 2 | Knowledge work with repeated local patterns, when host RAM and NVMe can hold the table | The table cannot live in 24 GB; it needs deterministic row prefetch overlapped with layer 1 compute | exact rows, stable async prefetch, no silent drop of the layer |
| Gated residual, 4 branches | Widens the residual stream with read and write gates | Stability and long range information flow at little arithmetic cost | Residual state can be stored in FP8, halving that state's bytes; budget 4 times hidden per KV slot | gate order, per-branch norm, no branch mixing |
| MTP, 1 layer | Drafts several tokens for verification | Low concurrency, structured or repetitive output | Saves wall time only when the target verification path is correct; adds state rollback surface | greedy and seeded sampling parity against non-spec BF16 |

The single most useful fact for this project is about QSA. Qwen's own report (arXiv 2608.30320 and the Flash-Next blog) says speedups begin at 64K and reach 7.6 times prefill and 4.9 times decode for the attention module at 1M. The same report says QSA matches or beats full attention on RULER and MRCR as context grows. On Ada, the win is KV memory and KV read traffic, not raw FLOPs. That is the lever for long context on a 24 GB card.

The sm_89 support picture matters as much as the architecture. A primary GitHub source, SGLang PR 36968, documents QSA sparse decode on real L20 and RTX 4090 sm_89 hardware: the FA4/CuTe path fails at the QSA head dimension 256, classic FA2 works, and without FA2 the server fails after loading weights. The PR adds a preflight check and is still open with failing CI as of late October 2026. A separate L20 MoE FP8 Triton config PR, 36993, was still in flight per the research base (T3, draft). vLLM's fast QSA and GDN paths target sm_90 and sm_121, so Ada falls back to Triton or FLA. llama.cpp PR 27742 is the only consumer GPU implementation with an accuracy table, and its closest hardware evidence is sm_86 4x3090. All of this matches `BUILD_SPEC.md` §3: on sm_89 we write our own sm_89 path for QSA and GDN and we do not inherit a Hopper or Blackwell kernel.

For MoE specifically, the research base (T3, draft) reports no single 4090 run: the smallest resident weight figure found is 41.73 GiB for a Q2 GGUF on a 128 GB unified machine, an IQ4_XS GGUF is about 93.7 GB, and the FP8 checkpoint is about 172.78 GiB. Consumer runs use 4x3090 or 128 GB unified memory. That is why dense is the first target in the build plan and sparse is second. It is a hardware fit argument, not a statement that MoE is worse.

## 3. Dense or sparse: the decision rule for sm_89

| Workload on the 4090 | Pick | Why |
|---|---|---|
| Interactive use, short or medium context, tight TTFT | dense 27B | Fits in VRAM, no router, no tier swaps, no indexer, fewest quality failure modes |
| Quality gate work, kernel numerics, new precision path | dense 27B first | Removes routing and residency as confounding variables; KLD deltas point at the kernel |
| Host RAM near 31 GB, or no fast NVMe | dense 27B | PLE table and expert tiers need tens of GB of host memory |
| Long context at 64K to 262K, agentic prefixes, repeated tool output | sparse, with QSA | Bounded retrieval budget keeps KV reads and attention cost from growing with depth |
| Work that needs more capacity than 24 GB can hold | sparse, tiered | MoE plus host and disk tiers trade latency for total model capacity |
| 1M context | sparse, only after validation | QSA is the architecture's answer to 1M, but depth sensitive quality is unmeasured here |

The two modes are complements on this hardware. Dense is the clean proving ground and the latency path. Sparse is the capacity and long context path. The phase order in `BUILD_SPEC.md` §5 already encodes this: P4 dense end to end before P5 sparse end to end.

## 4. The switch: how dense support actually fits this fork

The switch is a load time choice, not a per token mode. A model is selected once when the pack is opened, and every request after that uses the same path. This is already how the fork is built.

### 4.1 What exists today

`src/model/registry.cpp` keys adapters by `general.architecture` from the pack metadata. `adapter_for()` looks up that string and fails with the list of known names when nothing matches. `src/models/builtin.cpp` registers exactly two adapters, `qwen4exp` and `deepseek41`. Both are MoE.

The shared seam is `include/basalt/model/adapter.hpp`. `ModelDesc` already expresses the dense case: `n_expert`, `n_expert_used`, `n_expert_shared` and `n_ff_exp` are zero for a dense model, and `describe()` fills what the model has. `SideTableSpec` describes the PLE or engram table. `ModelSession` exposes one uniform decode, prefill, vision, checkpoint and probe API, including optional methods that return `NotImplemented` when a model has no drafter or no images. An adapter also declares `max_expert_cards()`, `quality_flags()`, `tokenizer_spec()`, `tool_form()` and the model's token names.

So the runtime does not need a dense/sparse branch in the hot path. A dense adapter registers its architecture name, returns zero expert fields, returns no side table unless its config has one, reports zero expert cards, and uses its own quality flags. The shared engine sees a smaller `ModelDesc` and does the same thing it already does for any model with no experts.

### 4.2 The constraint that shapes the work

`tools/check_model_layering.sh` enforces two rules: shared code never includes a model, and one model never includes another. A dense adapter therefore cannot include qwen4exp headers to borrow its GDN, attention or ViT code. The check also forbids a shared library from linking a model library.

Right now the architecture components live inside the sparse model. `src/models/qwen4exp/kernels/` holds the GDN recurrence, the QSA indexer and attention kernels, the gated residual kernels, the PLE kernels and the RoPE kernels. For example `dec_gdn_rec.cu`, `native_qsa_indexer.cu`, `native_qsa.cu`, `fused_gr.cu`, `hc_fused.cu`, `ple.cu` and `native_rope.cu`. The shared `include/basalt/kernels/` layer has low level bits such as bf16 helpers, dequant and sampler support, but not the architecture components.

This is the concrete gap behind the phrase "reuses the GDN and full attention machinery currently inside qwen4exp" in `BUILD_SPEC.md` §1.3. Reuse is not a header include. It requires promoting those components into the shared layer first.

### 4.3 The design that follows

1. Add one config driven `ArchSpec` built from field names present in the two verified configs: `layer_types`, `full_attention_interval`, head counts and head dims, `output_gate_type`, `hc_count` and `hc_lowrank`, `indexer_budget` and `indexer_compress_ratio`, `ple_layer_ids` and `ngram_*`, `mtp_*`, and the vision block. Both adapters consume this. A future Qwen4 checkpoint is a config and weight conversion change, not a new kernel port.
2. Promote the architecture components out of `src/models/qwen4exp/` into the shared kernel and component layer, parameterized by `ArchSpec`. That includes GDN recurrence with FP32 state, full attention with partial RoPE and MRoPE, QSA indexer and sparse core, gated residual, PLE lookup, MTP head and the ViT tower. Keep sm_89 replacements from `BUILD_SPEC.md` §3 inside these shared components, so both paths get them once.
3. Leave the adapter thin. The sparse adapter sets expert fields, tier policy, PLE side table and the QSA indexer from config. The dense adapter sets expert fields to zero, returns no expert tier or swap flags, and skips PLE if the config has no PLE fields. Both use the same GDN, attention, residual, MTP and vision code.
4. Keep quality flags per adapter. The sparse adapter needs `--adapt-swaps 0` and pinned tiers for identity runs; the dense adapter has no swaps to pin. Acceptance bands stay separate: dense gates cover state, conv, RoPE and accumulation, while sparse gates add router order, expert aggregation across tiers, indexer selection and PLE row identity.
5. Keep the sm_89 boundary absolute. Both adapters use the same sm_89 kernel map and the same one GPU at a time test discipline. No sm_86 fallback, no multi arch in v1, no sm_120 intrinsics on this path.

### 4.4 What the switch does not mean

It does not mean one checkpoint that flips between dense and MoE at runtime, and it does not mean a dense decoder inside the MoE session. The two are different weight layouts and different memory plans. The switch is the registry plus the shared component layer, and it happens before the first token.

## 5. What is still unverified

- Qwen4-27B has no weights, config or model card on 2026-11-16. Dense 27B with QSA is a convention and a T4 lead, not a verified target. The dense adapter target stays Qwen3.8-27B with its verified config.
- There is no controlled, same model, same corpus sm_89 KLD or PPL comparison between dense and sparse paths in public sources. The decision table in §3 is a systems argument and the project's own gate has to produce the numbers.
- SGLang PR 36968 is open, and Ada MoE kernel work is still landing. Their state needs a fresh check before any of it becomes a build assumption.
- 1M context quality is unmeasured. The 262K envelope is the one this project has local evidence for.
- The Qwen4-Max blog page is dated 2026/11/15 while reporting puts general availability on 2026-11-16. The difference does not change any engine decision.

## Sources

- Qwen team, Qwen4-Max blog: `https://qwen.ai/blog?id=qwen4-max` (primary vendor, fetched 2026-11-16)
- Qwen team, Flash-Next blog: `https://qwen.ai/blog?id=qwen3.8-flash-next` (primary vendor)
- Qwen team, Flash-Next tech report: arXiv 2608.30320, `https://arxiv.org/html/2608.30320v1` (primary paper)
- `Qwen/Qwen3.8-Flash-Next` config and model card on Hugging Face, verified 2026-11-16 and recorded in `BUILD_SPEC.md` §1.2
- SGLang PR 36968, QSA SM89 preflight with real L20 and RTX 4090 evidence: `https://github.com/sgl-project/sglang/pull/36968` (primary issue track, open)
- llama.cpp PR 27742, qwen4exp graph and accuracy table: `https://github.com/ggml-org/llama.cpp/pull/27742`
- local fork: `~/wt/basalt-ada-lift/src/model/registry.cpp`, `include/basalt/model/adapter.hpp`, `src/models/builtin.cpp`, `tools/check_model_layering.sh`, `src/models/qwen4exp/adapter.cpp`, `src/models/qwen4exp/kernels/`
- research base notes `doc:prep4qwen/qwen38-flash-next`, `doc:prep4qwen/qwen-official`, `doc:prep4qwen/targets` (all T3, status draft)
