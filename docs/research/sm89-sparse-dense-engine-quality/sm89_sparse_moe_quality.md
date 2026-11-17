# MoE / Sparse Execution on sm_89 (Ada) with BF16-Quality Preservation

Scope: quality-focused LLM inference for sparse Mixture-of-Experts (MoE) models on NVIDIA Ada (`sm_89`, e.g. RTX 4090 24 GB). BF16 is treated as the numerical reference where available; offload/tiering is evaluated primarily as a systems mechanism that should not change model math. Sources are a mix of primary local code/docs (basalt, the lab's Strata records) and upstream project documentation/papers fetched during this pass.

## How do MoE routing and expert dispatch affect numerical quality (router logits, top-k selection, expert aggregation)?

### Takeaway
Routing quality is determined by the router logits, the top-k/normalization policy, and the exact aggregation order/precision; moving expert weights between memory tiers is quality-neutral only if the same bytes/dtype reach the same kernel and aggregation path. For Qwen3-Next/Coder-Next-style MoE (512 experts, 10 activated, 1 shared), preserving router math and per-expert accumulation semantics is as important as preserving weights.

### Cited Findings
- Qwen3-Next-80B-A3B-Instruct uses a high-sparsity MoE with 512 experts, 10 activated experts, 1 shared expert, and expert intermediate dimension 512, in a hybrid layout with gated attention and Gated DeltaNet layers. — [Qwen3-Next-80B-A3B-Instruct README](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/README.md)
- Qwen3-Coder-Next reports the same MoE scale for its architecture (80B total / 3B activated, 512 experts, 10 activated, 1 shared, 48 layers, 256K native context) and is positioned as an efficient agentic coding model. — [Qwen3-Coder-Next README](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md)
- Qwen3-Next highlights Multi-Token Prediction (MTP) and a hybrid attention/MoE architecture as core design elements; MTP and routing interact with execution scheduling and can affect realized quality/performance tradeoffs if implementations diverge. — [Qwen3-Next-80B-A3B-Instruct README](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/README.md)
- Basalt’s expert dispatch pipeline separates residency/routing decisions from math: `ExpertCache` maps `(layer, expert)` to a VRAM slot or `kNotResident`, computes no expert itself, and the GPU computes held experts in grouped kernels/prompt path while the CPU pool computes the rest. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`
- Basalt’s `PeerExperts` is designed so an expert gives bit-identical rows on either card: “The peer computes with the SAME kernels as the primary's hit path (q8_1 activations + native_expert_grouped), from the same activation values…” — `~/wt/basalt-upstream-lift/include/basalt/core/peer_experts.hpp`
- Basalt explicitly treats the router-weight application contract as correctness-critical: the pool must not apply router weights if combination happens elsewhere (`moe_combine` sums `w[i] * parts[i]`), to avoid double-weighting. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`
- Basalt requires byte-for-byte verification that a cached VRAM slot contains the intended expert (`verify_slot` compares slot contents to the host blob), reflecting that plausible but wrong bytes produce plausible tokens. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`

### Inferences
- For BF16-reference quality on Ada, router logits computation, softmax/top-k selection, normalization (if used), expert accumulation order, and output cast/accumulator dtype must be identical across execution backends (GPU-resident, CPU-miss, peer GPU, disk-streamed into RAM). Any backend-specific fused kernel must be parity-checked against the reference path.
- Shared experts (as in Qwen3-Next/Coder-Next) and routed experts should be treated as distinct correctness domains: a dispatch bug that omits or double-applies one path can look like a small perplexity shift but a large task-level regression.
- Multi-device/heterogeneous dispatch (primary GPU + peer GPU + CPU) is compatible with numerical equivalence only when activations, weights, and reduction semantics are aligned; Basalt’s same-kernel/same-activation design is a concrete pattern to emulate.

### Gaps
- No primary source retrieved in this pass for the exact routing formula (e.g. top-k before/after softmax, renormalization, auxiliary/balance-loss behavior at inference, router dtype) of Qwen3-Next/Qwen3-Coder-Next/Qwen3.8-Flash-Next beyond architecture counts in model cards. The Qwen technical report/blog/PDF text was not successfully fetched as a full source.
- No measured BF16-vs-offload KLD/perplexity numbers for these specific Qwen MoE models on `sm_89` were found in the local research base or fetched sources.

## What are the expert offload patterns on sm_89 (VRAM→RAM→disk) and how do they preserve quality (no math change, only latency)?

### Takeaway
Quality-preserving offload treats memory tiers as placement/storage only: expert tensors are copied, not recomputed or requantized on transfer, and misses are executed by an equivalent math path (CPU, peer, or after staging). On 24 GB Ada cards, practical systems combine a hot expert cache in VRAM, a pinned/page-locked RAM resident set, and a file/SSD fallback for the cold tail.

### Cited Findings
- KTransformers states support for 3-layer (GPU-CPU-Disk) prefix cache reuse and heterogeneous CPU-GPU inference for MoE models, placing hot work on GPU and cold experts on CPU as an efficiency strategy. — [KTransformers README](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/README.md)
- Basalt’s `ExpertCache` is explicit VRAM slot storage/residency: it allocates device memory, fills slots from host arena, answers `(layer, expert) -> slot or -1`, and is designed so misses go to CPU while hits run on GPU; it verifies slots byte-for-byte after fill. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`
- Basalt’s `FileExpertSource`/arena design (per header comments and API) supports file-backed experts, pinned/page-locked complements, exchange buffers/spares for adaptive swaps, and DMA-friendly access for prompt/GPU paths, distinguishing resident RAM, file-backed fallback, and GPU-held experts. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`
- Basalt issues startup warnings and counters for experts that remain only on SSD (`spill_experts`, `spill_bytes`), making disk spill an explicit, observable tier rather than a silent fallback. — `~/wt/basalt-upstream-lift/include/basalt/program/expert_spill.hpp`
- `tier_profile.cpp` implements learned/adaptive expert placement between requests: it can save/load ranked profiles, move experts between slots/tiers to match a desired residency, and performs copies to restore or re-place experts (with synchronization), showing an operational VRAM↔RAM residency management loop. — `~/wt/basalt-upstream-lift/src/models/qwen4exp/serve/tier_profile.cpp`
- Strata 0.1.45 on the workstation (RTX 4090) ran Qwen3.8-Flash-Next UD-IQ4_XS with `--resident-budget-gib 40`, `--expert-cache auto`, `--pool-workers 15`, `--kv q4_0 --kv-resident 32768`; the run reported RAM 21.1M expert blobs and a file tier of 80,807 blobs / 382.8 GB read from GGUF in place with O_DIRECT (not page-cached), while KV block reads were 96.3% VRAM hits. — `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`
- The same Strata disk-tier measurement showed decode at ~218K depth reading ~9–14 MB/token from the file tier at ~36.6 MiB/s from SATA (disk ~35% busy), with decode 47–83 tok/s and 51–89% draft acceptance; NVMe A/B improved load and uncached-suffix prefill but not decode at 153K in that test (~66–73 tok/s across runs). — `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`
- Homelab Strata guidance documents the Ada/4090 operating point: IQ2_XS at 262K with q4_0 KV, ~15–16 GB VRAM plus ~20.9 GiB pinned RAM, 512K tight; it also warns that Strata refuses to start while anything holds VRAM and that heavy parallel work beside resident Strata can OOM/freeze due to pinned RAM pressure. — `~/<private>/guide/strata-and-ninfer.md`
- Basalt’s local design notes emphasize measuring held-out hit rate and actual free VRAM for sizing the expert tier, and that only a small share of `(layer, token)` pairs may have all selected experts resident (motivating split/hit-miss handling rather than assuming full-group residency). — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`

### Inferences
- On `sm_89`, VRAM→RAM→disk offload preserves BF16/reference quality when: (a) tier transitions copy exact expert bytes (or an intentionally chosen quantized expert format used consistently as the model’s reference), (b) execution uses numerically equivalent kernels for hit/miss paths, and (c) aggregation dtype/order is unchanged. Latency and throughput vary; logits need not.
- Pinned/registered host memory (as used conceptually in Basalt/Strata) is a quality enabler indirectly: it reduces staging complexity and enables predictable DMA/PCIe access paths, lowering the chance of ad hoc conversion paths that could diverge numerically.
- Disk tier quality risk is mostly operational/correctness (wrong file extent, stale mapping, partial read, misaligned blob) rather than mathematical; engines that use direct reads/O_DIRECT and explicit blob addressing must pair this with strong verification and accounting.

### Gaps
- No `sm_89`-specific kernel-level proof here that CPU-fallback expert math is bitwise-identical to GPU BF16/FP16 paths across all quant formats used in practice (Basalt states same kernels for primary/peer in that path, but CPU fallback equivalence depends on implementation/quant type).
- No published, citable latency/quality matrix for VRAM-only vs VRAM+RAM vs VRAM+RAM+NVMe on the same Ada system with fixed seeds and BF16 reference logits.

## What are the tiered memory architectures (basalt tiers, Strata, PolyStrata, KTransformers) and their quality implications?

### Takeaway
Tiered MoE architectures on consumer Ada GPUs are converging on “hot experts in fast memory, cold experts lower, same model math”: Basalt exposes explicit VRAM slot caches plus RAM/file/peer tiers and adaptive profiling; Strata demonstrates large pinned-RAM plus file-backed expert streaming for Qwen3.8-Flash-Next; KTransformers documents CPU-GPU heterogeneous inference and 3-layer GPU-CPU-disk prefix cache reuse. Quality implications hinge on byte-correct residency, consistent quant/compute paths, and stable routing/execution under swaps.

### Cited Findings
- **Basalt (local):** `ExpertCache` provides VRAM slot storage, residency tables, per-layer admission options, optional segmented/elastic VRAM management, and startup verification of slot contents; it is designed as storage/residency, not computation. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`
- **Basalt (local):** `PeerExperts` adds a second expert tier on another GPU, can share out ranked experts, supports arm/raise/fetch/scratch flows for prompt-time and expert movement, and includes sleep/wake and placement/reassignment operations for hot VRAM resize and residency management. — `~/wt/basalt-upstream-lift/include/basalt/core/peer_experts.hpp`
- **Basalt (local):** `ExpertSource`/`FileExpertSource` implement file-backed expert access, optional compact pinned complements of experts absent from GPU cache, exchange/spare mechanisms for swaps without losing the only RAM copy, top-up and home-region management when VRAM shrinks/grows, and tracking of RAM vs file reads/bytes. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`
- **Basalt (local):** `tier_profile.cpp` supports profile save/load/reset and rank-ordered placement of experts into main/peer slots, including draining in-flight swaps/commits before reconfiguration, which is relevant to deterministic residency transitions. — `~/wt/basalt-upstream-lift/src/models/qwen4exp/serve/tier_profile.cpp`
- **Strata (local ops/bench):** Strata serves MoE Flash-Next on 4090 with substantial pinned RAM and file-tier reads; at 512K context (0.1.39) it preserved recall across depths in reported checks and improved long-prompt read/decode over 0.1.38 in that A/B, with RAM headroom tuning affecting stability/headroom. — `~/<private>/evals/2026-10-04/strata-0139-512k/RESULTS.md`, `~/<private>/guide/strata-and-ninfer.md`
- **Strata disk tier (local bench):** Direct O_DIRECT file-tier expert reads from SATA/NVMe were measured; moving model data to NVMe improved load and uncached prefill markedly but did not materially change decode at 153K in that scenario, indicating tiering bottlenecks are workload/depth dependent. — `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`
- **KTransformers:** Project positions itself as CPU-GPU heterogeneous inference/fine-tuning for large models; README highlights 3-layer (GPU-CPU-Disk) prefix cache reuse and MoE-oriented placement (hot experts on GPU, cold on CPU) in its ecosystem/docs. — [KTransformers README](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/README.md)
- **PowerInfer (contextual tiering by activation locality):** PowerInfer exploits activation locality (“hot” neurons on GPU, “cold” on CPU) for fast local LLM serving; it targets sparse activation in neuron space rather than routed MoE experts, but demonstrates a similar hot/cold memory split principle on consumer GPUs. — [PowerInfer README](https://raw.githubusercontent.com/SJTU-IPADS/PowerInfer/main/README.md), [PowerInfer paper (arXiv:2312.12456)](https://arxiv.org/abs/2312.12456)

### Inferences
- For BF16-quality MoE inference, the safest tiering architecture is: immutable weight artifacts per tier, explicit residency tables, verified copies, and swap windows outside the math-critical token step (or fully synchronized). Basalt’s verify/fill/placement primitives and Strata’s measured recall/depth checks align with this principle.
- Prefix/KV tiering (GPU-CPU-disk, as in KTransformers) and expert tiering are orthogonal but interact on memory pressure; on 24 GB Ada, reclaiming VRAM from expert cache for KV (or vice versa) can force residency churn that must be controlled to avoid non-deterministic performance and hard-to-debug quality artifacts.

### Gaps
- **PolyStrata:** No primary documentation/source for PolyStrata was located in local workspace or successfully fetched from the web during this pass; its architecture and quality implications cannot be cited.
- **Basalt “tiers.cpp” filename:** Not found as a single file by that name in inspected trees; tiering behavior was inferred from `expert_cache.hpp`, `expert_source.hpp`, `peer_experts.hpp`, `expert_spill.hpp`, and `tier_profile.cpp` (local primary code).
- No KTransformers document fetched here that quantifies output-quality deltas (KLD/perplexity/task scores) attributable specifically to GPU-CPU-disk expert/KV tiering on Ada/sm_89.

## How do leading engines handle MoE quality on sm_89 (Strata, ik_llama.cpp, PowerInfer, FastLLM)?

### Takeaway
Direct, citable evidence for BF16-quality MoE handling *specifically on sm_89* is strongest in local Strata/basalt-adjacent operational records and weaker in fetched upstream docs for the other engines. Engines generally preserve quality by executing the same model weights/math across tiers (or by using validated quantized weights as the reference) while varying placement; the key differentiator is how explicitly they control residency, prefetch, and fallback paths.

### Cited Findings
- **Strata (sm_89/Ada 4090, local):** Used in production-like evals for Qwen3.8-Flash-Next on the workstation with large pinned RAM and optional disk/NVMe tiers; bench records report recall checks at long context (512K) and compare engine/build/config changes (headroom, storage backend), indicating an empirical quality/latency validation culture on Ada hardware. — `~/<private>/guide/strata-and-ninfer.md`, `~/<private>/evals/2026-10-04/strata-0139-512k/RESULTS.md`, `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`
- **ik_llama.cpp:** README advertises MoE-related features including dynamic expert cache and on-demand tensor reload, plus broad quantization/perf work; it also warns against certain flags/configurations (e.g. `-rtr` with hybrid CPU/GPU offload) that can force unintended CPU computation paths, which is a performance/correctness-footgun class relevant to quality validation. — [ik_llama.cpp README](https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/README.md)
- **PowerInfer:** Implements GPU-CPU hybrid inference based on activation locality (hot neurons preloaded on GPU, cold on CPU) to reduce transfers/memory demand while retaining model accuracy in its stated evaluation; this is neuron-sparsity, not routed MoE expert selection, but it is a leading consumer-GPU sparse inference design on RTX 4090-class hardware. — [PowerInfer README](https://raw.githubusercontent.com/SJTU-IPADS/PowerInfer/main/README.md), [PowerInfer paper (arXiv:2312.12456)](https://arxiv.org/abs/2312.12456)
- **FastLLM:** README describes CPU/GPU/NUMA/disk mixed inference for dense and MoE models, automatic deployment planning, and explicit MoE-related deployment controls in docs (e.g. expert cache/context settings referenced from README); it positions mixed-device execution as a core capability. — [FastLLM README](https://raw.githubusercontent.com/ztxz16/fastllm/master/README.md)
- **KTransformers (related leading heterogeneous engine):** Supports CPU-GPU hybrid inference and 3-layer prefix cache reuse, with MoE optimization focus in project messaging; useful comparator for tiered execution strategies. — [KTransformers README](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/README.md)

### Inferences
- On Ada 24 GB, “quality-safe” MoE engines typically pin a stable definition of the reference artifact (BF16 or a chosen quant, e.g. IQ2/IQ4/GGUF variant) and treat all residency moves as transport; engines that dynamically requantize, repack, or change kernel families per tier must provide parity evidence.
- Strata’s local results suggest that for very large MoE models on 4090, quality validation must be coupled to memory-headroom tuning (resident budgets, KV residency) because instability/OOM pressure can corrupt runs operationally even when math is intended to be identical.

### Gaps
- No fetched, engine-authored sm_89 benchmark tables comparing BF16-reference logits/perplexity to tiered/offloaded outputs for ik_llama.cpp, FastLLM, or KTransformers in this pass.
- PowerInfer evidence pertains to ReLU/neuron sparsity models and locality predictors on consumer GPUs, not routed MoE top-k expert dispatch quality on Qwen-style MoE architectures.
- FastLLM MoE-specific docs page (`docs/moe.md` in master) was not available at the fetched path (404), limiting citable detail on its expert cache/offload semantics.

## What are the quality risks in expert swapping (stale residency, race conditions, precision in copies)?

### Takeaway
Expert swapping risks are dominated by residency-bookkeeping errors and synchronization races (serving stale/wrong experts) and, secondarily, by precision/representation changes during copies or fallback execution. The mitigations visible in mature designs are explicit residency tables, synchronized swap windows, byte-level verification, and keeping a valid source of truth for every expert across tiers.

### Cited Findings
- Basalt’s `ExpertCache` enforces allocation checks, bounds-checked slot lookup, and byte-for-byte `verify_slot` comparison because “a slot table that is right about indices and wrong about bytes produces a plausible token…” — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`
- Basalt warns against caching quantized activations by pointer when the activation buffer address is reused each layer, as that can reuse layer-0 activations for later layers (“finite, plausible, completely wrong token”). While about activations (not expert swaps), it illustrates the class of stale-pointer/cache-key correctness bugs in MoE execution. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`
- Basalt’s tier/profile logic drains in-flight swaps/commits and synchronizes before reconfiguring residency (`apply_pending`, waits/synchronization paths) and constrains when placement operations may run (e.g. between requests/windows, no loan/pending states in certain flows), reducing race windows. — `~/wt/basalt-upstream-lift/src/models/qwen4exp/serve/tier_profile.cpp`, `~/wt/basalt-upstream-lift/include/basalt/core/peer_experts.hpp`
- Basalt’s peer/swap mechanisms include holding/reassign/exchange/spare-place logic so evicted experts are preserved (to RAM/free places) before slots are reused and incoming experts become resident only after copies land, aiming to keep every expert somewhere valid during swaps. — `~/wt/basalt-upstream-lift/include/basalt/core/peer_experts.hpp`, `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`
- Basalt’s `file_reads`/RAM-vs-file counters and explicit `spill_experts/spill_bytes` reporting make residency fallback observable; untracked fallback to file/mmap paths can mask both performance and correctness regressions. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`, `~/wt/basalt-upstream-lift/include/basalt/program/expert_spill.hpp`
- Strata’s 512K eval required raising resident headroom to avoid dangerously low available RAM under pinned allocations; memory pressure/overcommit conditions can create operational failures that may be confused with model-quality issues if not isolated. — `~/<private>/evals/2026-10-04/strata-0139-512k/RESULTS.md`

### Inferences
- **Stale residency:** the highest-risk bug is a residency map that points to a slot containing a different expert (after swap, shrink/grow, profile apply, or peer reassignment). This can produce coherent-looking but wrong outputs; byte verification and generation/versioning of placement state are essential.
- **Race conditions:** swaps concurrent with dispatch, prefill lending/return, or peer copies can read partially-updated plans/tables. Safe designs fence swaps at request/window boundaries and use stream/event ordering for all H2D/D2H/D2D moves.
- **Precision in copies:** raw byte copies of an existing quantized/blob format are precision-neutral; quality changes arise when a tier changes representation (e.g. dequant/requant, repack, different kernel dtype/accumulator, or FP16/BF16 cast points). On Ada, extra care is needed where CPU and GPU paths use different math libraries or vectorization.

### Gaps
- No incident/postmortem artifacts specifically documenting stale-expert or swap-race corruption in Strata/ik_llama/FastLLM/KTransformers were retrieved in this pass.
- No citable microbenchmark here quantifying numerical error introduced by PCIe copies vs in-place VRAM compute (expected near-zero for identical bytes, but implementation-specific).

## What validation is needed (perplexity per expert, routing stability, NIAH at long context)?

### Takeaway
A quality-focused sm_89 MoE engine needs layered validation: bit/Logit/KLD parity for deterministic cases, aggregate perplexity and task evals versus a BF16 (or declared reference) baseline, routing/residency telemetry under fixed prompts, and long-context stress tests (including NIAH-style retrieval) across tiering modes and memory-pressure regimes.

### Cited Findings
- Basalt’s project guidance (AGENTS/CONTRIBUTING excerpts in repo) requires parity/correctness checks and careful A/B discipline (pin tiers/conditions, alternate builds, report odd results), reflecting sensitivity of measurements to placement drift and session differences. — `~/wt/basalt-upstream-lift/AGENTS.md`, `~/wt/basalt-upstream-lift/CONTRIBUTING.md` (present in local clone)
- Basalt tooling mentions pinning expert tiers/adaptive swaps off for greedy identity/PPL gating and alternating builds (ABBA) to avoid drift from auto-sized tiers and session variance. — Local grep evidence from `tools/greedy_identity.py`, `tools/ppl_gate.py` in `~/wt/basalt-upstream-lift` (e.g. `--adapt-swaps 0`, pin tiers when VRAM use differs)
- Strata long-context evals on 4090 included depth/recall checks at 512K and compared prefill/decode under different configs/headroom/storage backends, demonstrating the need to validate both correctness signals (recall) and performance under realistic tier pressure. — `~/<private>/evals/2026-10-04/strata-0139-512k/RESULTS.md`, `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`
- Basalt exposes/maintains observability hooks relevant to validation: residency/slot verification, spill counters, RAM/file read counters, and profile/adaptation behavior (`tier_profile`, `expert_spill`, `expert_source` counters). — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`, `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`, `~/wt/basalt-upstream-lift/include/basalt/program/expert_spill.hpp`

### Inferences
- **Per-expert/per-layer validation (practical):** full “perplexity per expert” is not usually a standard end metric, but engines should log selected expert IDs/weights, hit/miss tier, and per-layer dispatch counts for fixed prompts to detect routing instability or residency skew. When a parity failure occurs, isolate by freezing placement/profile and comparing expert outputs for the routed set.
- **Routing stability:** run deterministic greedy decode with fixed seed/temperature=0, fixed placement/profile, and compare token streams/logits; then run controlled adaptation experiments to ensure swaps do not alter outputs when semantics should be invariant.
- **Long-context/NIAH:** on 24 GB Ada, validate at target context lengths with KV quant/offload settings identical across runs; include needle-in-a-haystack/retrieval probes at multiple depths and after cache eviction/swap events, because KV and expert tiering interact under memory pressure.
- **BF16 reference protocol:** define a reference configuration (e.g. BF16 weights/activations where feasible, or highest-fidelity available on 4090), capture logits/perplexity on a fixed corpus slice, and require bounded KLD/perplexity deltas for offloaded/quantized/tiered modes.

### Gaps
- No standard, cited NIAH benchmark results for Strata/ik_llama/FastLLM/KTransformers on sm_89 were retrieved in this pass.
- No end-to-end BF16-reference validation suite specification (exact datasets, KLD thresholds, acceptance criteria) was found in local docs for the basalt-ada-engine project beyond general guidance.

## Papers/repos on HuggingFace or Chinese sources for MoE quality on Ada

### Takeaway
This pass found strong citable materials on heterogeneous/offloaded inference (KTransformers, FastLLM, PowerInfer) and local Ada-4090 operational evidence (Strata), but limited direct papers/repos explicitly measuring *MoE output-quality preservation on sm_89/Ada*. Several likely-relevant Chinese/community sources could not be reliably retrieved or verified here.

### Cited Findings
- **KTransformers (GitHub, China-origin project, SJTU/MADSys ecosystem):** open-source framework for CPU-GPU heterogeneous inference/fine-tuning of large MoE models, with 3-layer prefix cache reuse and MoE optimization claims. — [KTransformers README](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/README.md), [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)
- **FastLLM (GitHub, Chinese documentation/project):** high-performance inference engine supporting dense/MoE models with GPU/CPU/NUMA/disk mixed inference and MoE-focused deployment guides referenced from README. — [FastLLM README](https://raw.githubusercontent.com/ztxz16/fastllm/master/README.md), [ztxz16/fastllm](https://github.com/ztxz16/fastllm)
- **PowerInfer (paper/repo):** SOSP 2024 paper and implementation for consumer-GPU sparse inference via activation locality (hot/cold neurons), evaluated on RTX 4090; relevant to sparse/offload quality-preservation discipline though not routed MoE. — [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU (arXiv:2312.12456)](https://arxiv.org/abs/2312.12456), [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)
- **Qwen3-Next / Qwen3-Coder-Next (Hugging Face):** official model cards documenting MoE architecture parameters (512 experts, 10 active, 1 shared) and deployment guidance across vLLM/SGLang/llama.cpp/KTransformers ecosystem. — [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/README.md), [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md)
- **ik_llama.cpp (GitHub):** llama.cpp fork with MoE features including dynamic expert cache/on-demand tensor reload and extensive quantization/backends; commonly used in local/consumer-GPU MoE experimentation. — [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **Local basalt/Strata evidence (Ada 4090):** operational source/docs and eval records in this workspace provide concrete sm_89 tiering/offload behavior and validation signals for MoE workloads. — `~/wt/basalt-upstream-lift/include/basalt/core/expert_cache.hpp`, `~/wt/basalt-upstream-lift/include/basalt/core/expert_source.hpp`, `~/<private>/evals/2026-10-04/strata-0139-512k/RESULTS.md`, `~/<private>/evals/2026-10-18/strata-disk-tier/RESULTS.md`

### Inferences
- Chinese/open-source inference engines (KTransformers, FastLLM) are high-value primary implementation references for MoE offload mechanics on commodity hardware; however, for a BF16-quality claim on sm_89, one must supplement them with reproducible parity/perplexity/NIAH measurements on the target Ada GPU and target model/quant.
- Hugging Face model cards are necessary for architecture/serving parameters but insufficient for numerical-quality guarantees under tiered execution; engine-level validation artifacts are required.

### Gaps
- **PolyStrata:** requested in scope but no citable primary repo/doc found.
- **Qwen3.8-Flash-Next technical report:** explicitly requested, but no full technical report text was retrieved in this pass (model naming/cards and local Strata usage evidence were found instead).
- No retrieved papers specifically titled/focused on “MoE quality preservation on Ada/sm_89” with KLD/perplexity/task deltas across VRAM/RAM/disk tiers.
- Web search/fetch coverage was limited in this environment (multiple searches returned no results; some docs/URLs unavailable), so additional Chinese sources (e.g. Zhihu/blog posts, ModelScope pages, CN engine docs/forums) remain uncollected and should be gathered in a follow-up pass with working search and targeted queries (KTransformers/PowerInfer/FastLLM technical blogs, Qwen tech reports, MoE offload correctness on RTX 4090). 
