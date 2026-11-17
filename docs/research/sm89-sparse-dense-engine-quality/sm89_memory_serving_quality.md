# sm_89 Memory, Offload, and Serving Quality Under Pressure (BF16 Reference)

## How does host RAM pressure affect quality (swapping, pinned memory, OOM)?

### Takeaway
On Ada/sm_89 hosts, host RAM pressure threatens *availability and correctness* more than steady-state numeric quality when BF16 compute/KV are preserved: pinned/page-locked expert or staging buffers cannot be swapped (so they trade OOM risk for predictability), ordinary pageable allocations can induce kernel swapping that causes severe latency spikes/timeouts and watchdog-like failures, and allocator/driver fragmentation or co-tenants can turn a “fits in VRAM” configuration into startup or runtime OOM/preemption. Quality loss usually appears indirectly (truncated/failed requests, recomputation after preemption, disabled caches/features, or forced lower precision/capacity) rather than from BF16 math changing.

### Cited Findings
- Strata on the workstation pages/locks ~20.9–21 GiB of host RAM for experts and leaves ~10 GiB on a 31 GB box; a parallel CUDA build at high parallelism (`-j 16`) next to a resident Strata OOM-froze the machine, so maintainers must stop Strata before compiling or build with low concurrency while watching `MemAvailable`. — `~/<private>/guide/strata-and-ninfer.md`
- Strata 0.1.39 pinned about ~2 GiB more RAM at 512K context than 0.1.38; without `STRATA_RESIDENT_HEADROOM_GIB=6` free RAM fell to 1.4 GiB at a 477K prompt, motivating explicit resident headroom to avoid host-memory exhaustion. — `~/<private>/guide/strata-and-ninfer.md`
- NInfer treats `memory_headroom` (default 512 MiB, per-rank planning allowance) and `context_capacity` as startup/planning controls; each rank deducts headroom from memory available after initialization, so explicit headroom/capacity changes the admitted KV/context budget rather than silently overcommitting. — `~/wt/ninfer-qwen4/ninfer.example.toml`, `~/wt/ninfer-qwen4/docs/startup-configuration.md`
- NInfer’s pending queue is bounded (`max_pending_requests`, default 16) with a pre-execution deadline (`pending_timeout`, default 30 s, not a total generation deadline), limiting backlog amplification under pressure. — `~/wt/ninfer-qwen4/ninfer.example.toml`
- vLLM sizes KV cache against measured free memory and warns that preemption/recompute occurs when KV space is insufficient; mitigation includes leaving/increasing `gpu_memory_utilization` (or avoiding caps that reduce KV), decreasing concurrency/batch tokens, or adding parallelism. Preemption affects end-to-end latency and can force recomputation work. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
- vLLM’s extensible KV cache (CUDA/ROCm default behavior in current docs) measures after warmup/graph capture and keeps a safety margin; maintainers are advised to set `gpu_memory_utilization` below 1.0 only to leave room for memory vLLM does not measure (other processes, unified memory, weight-update buffers, connector buffers). Co-tenants/late allocations are a stated OOM risk because measurement is a snapshot. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
- `--kv-cache-memory-bytes` can pin KV allocation and skip profiling/graph-estimation steps, but an optimistic value can fail at allocation and a conservative value caps concurrency; the value is only valid for the same GPU and initial free memory. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html), [Particula: vLLM KV Cache OOM](https://particula.tech/blog/vllm-kv-cache-oom-debugging)
- Pinned host allocations used by CPU offload paths can round up significantly (reported ~1.9x in some tensor shapes in issue discussion), creating host-RAM surprises distinct from requested bytes. — [Particula: vLLM KV Cache OOM](https://particula.tech/blog/vllm-kv-cache-oom-debugging)

### Inferences
- For BF16-reference serving on a 24 GB Ada card with host-resident experts/staging (Strata-like), the primary quality-preservation control is *admission/headroom discipline*: reserve VRAM/RAM for measured peaks, avoid concurrent memory-heavy jobs, and prefer rejecting/queuing with deadlines over entering swap/OOM states that force restarts or dropped work.
- Swapping pageable inference state is generally incompatible with stable serving quality because tail latency and timeouts can cascade into retries, partial outputs, or engine restarts; pinned memory avoids swap but makes capacity planning stricter and failures more abrupt (OOM/refusal).

### Gaps
- No public, sm_89/Ada-specific microbenchmarks were found in gathered sources quantifying BF16 output drift caused solely by host swapping vs pinned-memory pressure; most evidence is operational (OOM, stalls, preemption) rather than bitwise-quality measurements.
- Exact OS-level swap behavior and NUMA effects on the workstation’s specific kernel/systemd configuration were not directly measured here.

## What are the offload patterns (VRAM→RAM→disk) and their quality implications (none, if done correctly)?

### Takeaway
Correctly implemented offload/tiering is usually *capacity-latency* engineering, not a numeric-precision change: keeping BF16 weights/KV/activations bit-identical while moving cold data across VRAM/RAM/NVMe preserves answer quality but can increase TTFT/ITL, trigger recomputation/eviction policies, or expose correctness bugs (fragmentation, stale blocks, race conditions). Quality risk rises when offload is coupled with quantized KV, lossy eviction, or incomplete cache invalidation.

### Cited Findings
- vLLM preemption in V1 defaults to recompute rather than swap; when KV space is insufficient, requests may be preempted and recomputed later, trading latency/throughput for correctness of state reconstruction (no “swapped KV” semantics by default). — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html), [Particula: vLLM KV Cache OOM](https://particula.tech/blog/vllm-kv-cache-oom-debugging)
- vLLM supports CPU offload paths/flags in ecosystem usage (e.g. `--cpu-offload-gb`, KV offloading backends/integrations discussed in community/docs-adjacent sources), but offloading model parameters is distinct from KV-cache pressure remedies and can carry large performance costs and version-specific behavior caveats. — [Particula: vLLM KV Cache OOM](https://particula.tech/blog/vllm-kv-cache-oom-debugging)
- LMCache-based integrations enable KV movement/reuse across tiers (GPU/CPU/remote storage backends in broader ecosystem), with configuration knobs for chunking, async loading, and backend behavior; this is aimed at reuse/latency/capacity rather than changing model math when values are preserved. — [Ceph blog: KV Caching with vLLM, LMCache, and Ceph](https://ceph.io/en/news/blog/2025/vllm-kv-caching/)
- TurboMind/LMDeploy KV cache manager treats device memory as pooled slots with LRU eviction and can convert evicted sequences to compact token-ID history for later cache-miss reconstruction (decode context again), presenting “infinite” effective device memory from the maintainer perspective while changing recompute cost, not intended semantics. — [LMDeploy TurboMind Architecture](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)
- SGLang exposes memory/scheduling controls (e.g. `--mem-fraction-static`, chunked prefill, radix/prefix caching-related behavior) and supports deterministic mode under specific backends/configurations; cache/eviction policy interacts with reuse and reproducibility guarantees. — [SGLang Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html), [SGLang Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- A reported vLLM issue describes severe GPU memory leak/OOM with native KV offloading in certain multi-node PP/TP setups, disappearing when offloading is disabled, indicating offload implementations can introduce correctness/resource bugs under load. — [vLLM issue #44181](https://github.com/vllm-project/vllm/issues/44181)
- KV cache compression/quantization literature and vendor guidance distinguish memory-saving representations (e.g. FP8/INT variants) from exact copies; compressed/quantized KV changes numerical representation and therefore can affect quality, whereas byte-faithful movement does not. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html), [NVIDIA Research: KV Cache Compression and Its Infra Problems](https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/)

### Inferences
- For a BF16-reference quality target, prefer *exact* tiering (BF16 weights/KV/state copied unchanged, or recompute from exact token history) over quantized/offloaded KV formats; treat KV quantization/offload-compression as a separate quality trade study.
- Disk-tier or remote-tier KV should be validated as a cache/reuse optimization with explicit eviction and corruption checks; “done correctly” means strong identity/invalidation semantics, checksums/versioning where possible, and no silent reuse of stale blocks.

### Gaps
- Direct, comparable BF16-fidelity measurements for VRAM↔RAM↔NVMe KV offload on sm_89 (same prompts/seeds, long context, concurrent load) were not found in gathered sources.
- Basalt’s intended offload/tiering design for the `basalt-ada-engine` target is not documented in the files inspected (`~/<private>/projects/active/basalt-ada-engine/README.md` missing); implementation-specific guarantees cannot be cited.

## How do leading engines handle memory under pressure (Strata OOM risk, NInfer headroom guards, basalt tiers)?

### Takeaway
Observed patterns favor *bounded admission, explicit headroom, and recoverable lifecycle states* over best-effort overallocation: Strata relies on large pinned host residency plus maintainer headroom knobs and manual isolation from competing workloads; NInfer exposes explicit memory/capacity/headroom and queue/time budgeting; basalt’s server layer shows practical serving guards (VRAM availability checks before load, idle sleep/unload, queueing/serialization primitives, host-RAM checks before sleep) that reduce pressure but require engine-level KV/memory policies for full correctness.

### Cited Findings
- Strata production guidance: unload other VRAM holders before start, expect ~15–16 GB VRAM plus ~20.9 GiB pinned RAM for the profiled configuration, stop Strata before heavy builds or constrain build parallelism to avoid OOM/freezes, and use `STRATA_RESIDENT_HEADROOM_GIB` to retain free RAM margin at large contexts. — `~/<private>/guide/strata-and-ninfer.md`
- NInfer config/docs expose per-rank planning controls: `context_capacity`, `memory_headroom` (deducted per rank), `prefill_chunk`, `cuda_graph`, `max_concurrency`, `max_pending_requests`, and `pending_timeout`; startup/deployment docs describe headroom/capacity interaction and host-capacity accounting for context cache features. — `~/wt/ninfer-qwen4/ninfer.example.toml`, `~/wt/ninfer-qwen4/docs/startup-configuration.md`, `~/wt/ninfer-qwen4/docs/cli.md`
- Basalt serve (`Service`) tracks load/wake state, refuses/defers load when insufficient free VRAM (`min_free_vram_mib`) with a wait window, runs an optional `before_load` hook, supports idle unload and idle sleep (releasing VRAM while keeping process/RAM state) with a host-RAM fit check and margin (`SLEEP_RAM_MARGIN_MIB`), and maintains FIFO/control locks plus queued/running bookkeeping. — `~/wt/basalt-upstream-lift/serve/service.py`
- Basalt server startup/configuration includes lazy loading, swap/start retry behavior for multi-model serving (`SWAP_START_TRIES`), and warns on tight RAM relative to arena usage; it can run multiple model services behind a router while serializing model residency/load transitions. — `~/wt/basalt-upstream-lift/serve/server.py`
- vLLM documents memory tuning under pressure via `gpu_memory_utilization`/`--kv-cache-memory-bytes`, reducing `max_num_seqs`/`max_num_batched_tokens`, and parallelism changes; warns that co-tenants or post-measurement allocations can invalidate sizing assumptions. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
- LMDeploy TurboMind uses a pooled KV cache manager with LRU eviction and cache-hit/miss behavior (evicted sequences can be compacted and later reconstructed), aiming to absorb pressure without surfacing hard failures to users. — [LMDeploy TurboMind Architecture](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)

### Inferences
- A BF16-quality-first sm_89 engine should combine: (1) conservative static reservations (weights + activations + graphs + KV + non-torch/driver slack), (2) bounded queues with deadlines, (3) explicit preemption/recompute or admission control policy, and (4) observability that distinguishes VRAM, pinned host RAM, pageable host RAM, and cache tiers.
- Strata’s pinned-RAM model is effective for expert residency/latency but shifts failure mode to host-level contention; NInfer’s config surface is better suited to predictable multi-request admission; basalt’s lifecycle controls are useful building blocks for “pressure without corruption” if paired with robust KV ownership/invalidation.

### Gaps
- No authoritative “basalt tiers” memory policy document was found in local basalt-ada-engine paths searched; details of VRAM↔RAM↔disk tiering, eviction, and recovery semantics for the target engine remain unspecified.
- Public side-by-side pressure behavior (OOM recovery latency, success rate, quality drift) for Strata/NInfer/basalt on the same Ada workload was not available.

## What concurrency patterns preserve quality (batching, scheduling, determinism under load)?

### Takeaway
Quality-preserving concurrency on Ada prioritizes *deterministic scheduling modes, bounded batch/token budgets, and cache policies with explicit correctness contracts* over maximal throughput: continuous/chunked batching is compatible with BF16 quality when kernels and reductions are numerically stable, but reproducibility can still vary with batch composition, radix/prefix-cache behavior, floating-point reduction order, and speculative/async execution unless explicitly controlled.

### Cited Findings
- vLLM V1 uses token-budget-based scheduling with chunked prefill enabled by default when possible; `max_num_batched_tokens` trades TTFT vs ITL and affects batching/composition, while preemption/recompute is the default pressure response. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html), [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
- vLLM supports async scheduling that overlaps CPU scheduling with GPU execution using placeholders, which improves utilization but adds ordering/state complexity that must be handled carefully during preemption/reset scenarios. — [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
- SGLang provides deterministic inference mode (`--enable-deterministic-inference`) with backend-specific compatibility constraints (FlashInfer/FA3/Triton), and notes that varying batch sizes and floating-point non-associativity can cause different outputs; radix cache behavior and feature support differ by backend in deterministic mode. — [SGLang Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- SGLang server arguments expose fine-grained scheduling/memory controls (`--max-running-requests`, `--max-queued-requests`, `--max-total-tokens`, `--chunked-prefill-size`, `--schedule-policy`, radix eviction policies, unified memory/page-major options), enabling maintainers to cap pressure and shape batching. — [SGLang Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html)
- NInfer exposes bounded concurrency/admission controls (`max_concurrency`, `max_pending_requests`, `pending_timeout`) and prefix-reuse/checkpoint controls, which constrain in-flight work and context-cache behavior under load. — `~/wt/ninfer-qwen4/ninfer.example.toml`
- LMDeploy emphasizes persistent/continuous batching and blocked KV cache management for throughput under concurrency; TurboMind’s cache manager reuses/evicts/reconstructs sequence state based on hits/misses. — [LMDeploy README](https://github.com/InternLM/lmdeploy), [LMDeploy TurboMind Architecture](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)

### Inferences
- To preserve BF16 answer quality under concurrency, fix as many sources of run-to-run variability as feasible: deterministic kernels/backend mode, stable batching policy or batch-size constraints, fixed sampling seeds/params, and cache modes with proven equivalence; separately validate that prefix/radix cache hits are semantically identical to misses.
- Admission control (queue caps + deadlines + max in-flight tokens/requests) is a quality mechanism as well as a latency mechanism because it prevents overload regimes where engines preempt, thrash, or hit error paths.

### Gaps
- No engine-comparable matrix was found showing BF16 determinism/quality deltas on sm_89 across specific concurrency levels for vLLM, SGLang, LMDeploy, NInfer, and Strata with identical prompts and decoding settings.
- SGLang deterministic mode backend coverage on Ada/sm_89 for all relevant model architectures in this workspace was not verified from local artifacts.

## What are the serving quality risks (non-determinism, precision in batching, KV cache corruption)?

### Takeaway
The dominant serving-quality risks under pressure are *nondeterminism and state-integrity failures*, not BF16 arithmetic drift in isolation: batch-size/reduction-order effects, cache reuse bugs, scheduler/preemption races, and offload/copy errors can produce different or invalid outputs. Precision risks usually enter via optional quantized KV/FP8 paths, aggressive memory-saving modes, or implementation bugs rather than canonical BF16 execution.

### Cited Findings
- SGLang documents floating-point non-associativity and varying batch composition as causes of nondeterminism even at `temperature=0`, and provides a deterministic inference mode with backend/feature constraints to improve reproducibility. — [SGLang Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- vLLM’s performance/memory guidance highlights preemption/recompute behavior, CUDA-graph/compile/cache interactions, and measured-vs-pinned memory sizing tradeoffs; these affect latency and sometimes feasibility/stability, and can interact with reproducibility depending on configuration. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
- A reported vLLM native KV offloading bug in specific topologies caused monotonic GPU memory growth/leak and eventual OOM under load, indicating cache/offload path correctness risks that can manifest as service failure or corrupted runtime state. — [vLLM issue #44181](https://github.com/vllm-project/vllm/issues/44181)
- vLLM async scheduling introduces placeholder-based overlap and special handling around preemption/reset; correctness depends on scheduler invariants when requests are preempted/resumed or prefix state changes. — [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
- LMDeploy TurboMind reconstructs evicted KV state from compact token history on cache miss, which is correctness-preserving only if token history, sequence identity, and decoding semantics remain consistent. — [LMDeploy TurboMind Architecture](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)
- Basalt serve code includes engine death detection/restart paths, vision subprocess health checks, and explicit refusal paths for unsupported/unsafe requests, reflecting operational risks (engine crash, partial subsystem failure) that can degrade quality/availability if not surfaced cleanly. — `~/wt/basalt-upstream-lift/serve/service.py`

### Inferences
- For BF16-reference claims, “precision in batching” should be treated as a reproducibility contract: same model/weights/dtype/sampling may still differ under different batch shapes unless deterministic kernels and cache semantics are enforced and validated.
- KV-cache corruption is most likely at boundaries (offload copies, eviction/reuse, paging/block tables, async handoff, restart/resume). Strong request/sequence IDs, generation counters, and cache invalidation on engine restart are essential controls.

### Gaps
- No public postmortem or reproducible test case was collected here proving silent KV-cache content corruption (wrong outputs without crash) on sm_89 for the engines surveyed; available evidence is stronger for leaks/OOM/nondeterminism than for silent corruption.
- Chinese-language sources (blogs/forums/papers/repos) specifically addressing BF16 serving quality under memory pressure on Ada/sm_89 were not identified in the searches performed.

## What validation is needed (stress tests, OOM recovery, quality under load)?

### Takeaway
Validation must cover *correctness, reproducibility, and recovery* at the same time: run BF16-golden equivalence tests across cache hit/miss and batching modes, then stress host/VRAM pressure with controlled co-tenancy and overload to verify graceful degradation (queueing, preemption, admission refusal) rather than corruption. Recovery tests should prove engines restart cleanly and never reuse stale KV/state.

### Cited Findings
- vLLM recommends operational tuning/observability around preemption, memory sizing, and startup profiling; it also provides mechanisms (`--kv-cache-memory-bytes`, graph/cache controls) whose correctness depends on matching hardware/free-memory conditions and should be regression-tested after environment changes. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html)
- SGLang provides deterministic test modes (single, prefix, radix cache consistency) to verify identical outputs under varying batch/prefix conditions when deterministic inference is enabled. — [SGLang Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- Strata operational experience shows host-RAM headroom and isolation from concurrent heavy jobs are required to avoid OOM/freezes at large contexts, implying stress tests must include co-tenant CPU/GPU/build workloads and long-context prompt ingestion. — `~/<private>/guide/strata-and-ninfer.md`
- NInfer’s bounded pending queue/timeouts and explicit memory/capacity settings create testable contracts for overload behavior (reject/timeout vs stall) and startup fit. — `~/wt/ninfer-qwen4/ninfer.example.toml`, `~/wt/ninfer-qwen4/docs/startup-configuration.md`
- Basalt serve implements load/wake/sleep/unload/restart and VRAM/RAM gating paths that should be exercised with fault injection (engine death, encoder death, low-VRAM, low-RAM, idle transitions) to confirm no stale state leaks across lifecycles. — `~/wt/basalt-upstream-lift/serve/service.py`, `~/wt/basalt-upstream-lift/serve/server.py`

### Inferences
- A minimal BF16-quality validation suite for sm_89 should include: (a) golden-output parity vs reference decode (single request, greedy and fixed-seed sampling), (b) cache equivalence (prefix hit vs miss, warm vs cold), (c) concurrency invariance tests (N=1..max, varied arrival/chunking) in deterministic mode where supported, (d) pressure tests (near-capacity KV, induced preemption/eviction, host-RAM contention), and (e) recovery tests (kill/restart during prefill/decode, OOM injection, verify no cross-request cache reuse).
- Record both quality and systems metrics per run: exact text/token hashes (when deterministic), logprob drift tolerances (when non-deterministic), TTFT/ITL, preemption/eviction counts, queue wait/timeout counts, RSS/VRAM/pinned-memory peaks, and restart reason codes.

### Gaps
- No canonical, published BF16 acceptance thresholds (e.g. allowed logprob delta under batching/load on Ada) were found for these engines; thresholds must be defined by the project’s BF16 reference policy.
- End-to-end stress harness results for the target `basalt-ada-engine` repository were not available in the inspected workspace.

## Papers/repos on HuggingFace or Chinese sources for memory/serving quality on Ada

### Takeaway
High-signal public material found is mostly *engine documentation, architecture notes, and issue trackers* in English; Ada/sm_89-specific, BF16-serving-quality-under-memory-pressure sources on HuggingFace papers or Chinese venues were sparse in this pass. The most relevant primary repos/docs are vLLM, SGLang, LMDeploy, and local engine artifacts (Strata/NInfer/basalt).

### Cited Findings
- vLLM optimization and architecture docs describe KV-cache sizing, preemption, chunked/prefix/async scheduling, and memory controls relevant to pressure behavior. — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization.html), [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
- SGLang documents deterministic inference, backend compatibility, and extensive server memory/scheduling arguments relevant to quality/reproducibility under load. — [SGLang Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html), [SGLang Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html)
- LMDeploy/TurboMind document persistent batching and KV-cache management (blocked/pooled cache, eviction/reconstruction) and position high-throughput inference features. — [LMDeploy TurboMind Architecture](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html), [LMDeploy GitHub](https://github.com/InternLM/lmdeploy)
- Local operational and implementation sources capture Ada/4090 deployment realities: Strata’s pinned-RAM/OOM/headroom behavior, NInfer’s memory/concurrency config surface, and basalt’s serving lifecycle/guard logic. — `~/<private>/guide/strata-and-ninfer.md`, `~/wt/ninfer-qwen4/ninfer.example.toml`, `~/wt/ninfer-qwen4/docs/startup-configuration.md`, `~/wt/basalt-upstream-lift/serve/service.py`
- A concrete offload-path correctness issue is tracked publicly for vLLM (native KV offloading leak/OOM in specific topology). — [vLLM issue #44181](https://github.com/vllm-project/vllm/issues/44181)

### Inferences
- For sm_89/Ada, vendor-neutral engine docs plus local deployment records currently provide more actionable quality-under-pressure guidance than easily discoverable academic/HF-paper sources; implementation details and maintainer guardrails matter as much as algorithmic claims.

### Gaps
- Targeted HuggingFace paper/discussion results specific to “Ada sm_89 BF16 serving quality under memory pressure/offload/determinism” were not retrieved in this session.
- Targeted Chinese-language papers/repos/blogs (CNKI/arXiv-zh mirrors/知乎/微信公众号/Gitee/HF CN discussions) on this exact topic were not retrieved; dedicated Chinese-source search remains open.
- Dynamic Split&Fuse internals in LMDeploy were referenced at a high level in project descriptions encountered during search, but a citable primary technical document detailing its memory/quality behavior under pressure was not captured beyond general LMDeploy/TurboMind materials above.
