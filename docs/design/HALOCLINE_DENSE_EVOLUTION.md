# Dense evolution path (plan of record, opened 2026-11-18)

> [!NOTE]
> Status 2026-11-18: phase 1 (repository audit) in progress. Nothing in this page is a measured claim unless it names an
> `evals/` directory. Status words: **implemented** (on main, gated), **experimental** (opt-in flag or branch),
> **planned** (written down, not built), **unverified** (built, not measured on the GPU).

The maintainer's brief is kept verbatim at the end of this page. This section records how it is being carried out and
what already constrains it.

## Decisions that bound the path

| Decision | Where | Effect on this path |
|---|---|---|
| Dense weights are Unsloth Dynamic v3.1 GGUFs, repacked losslessly, no requantization | roadmap D1 | the pack copies Unsloth's tensors as stored; a lossless re-layout of the same values is allowed when the profile shows the GEMV is the bottleneck |
| UD-Q4_K_XL is the dense quant | roadmap D8 item 3 | the reference target of section 4 is Qwen3.8-27B UD-Q4_K_XL |
| K8V4 is the served dense KV; the engine default for an unset `--kv` stays fp16 | roadmap D8, 2026-11-18 | KV changes are gated at depth (65,536 and 140,288), never at short context alone |
| sm_89, one card | `SPECS.md`, `CMakeLists.txt` | no other architecture and no multi-GPU path (issue #2, closed not planned 2026-11-18) |
| Every lever comes from cited research and runs both paths' gates when it touches shared code | roadmap, standing orders | the sparse path's gates run on any change to `src/kernels/` or the shared serve code |
| A decode-kernel change may alter summation order if fp64 parity and the paired depth KLD hold | maintainer, 2026-11-18 | applies to the S1 decode kernel work below |

**Open (asked 2026-11-18):** section 2's weight progression (BF16, FP8, NVFP4, INT8, groupwise integer) against D1. The
proposal on the table: D1 stays for served weights; FP8, NVFP4 and INT8 weights are research arms only, each served
only after a measured KLD win at depth over UD-Q4_K_XL, with BF16 as the reference through the frozen KLD bases. NInfer's
requantized weights measured about 10 times further from BF16 than Halocline's repack on target-token dNLL
(`evals/2026-11-18/ninfer-depth-ppl`), which is the evidence behind D1.

## Phases

| Phase | Status | Where |
|---|---|---|
| 1. Repository audit (capability map) | in progress | the map lands in this page when its references are checked |
| 2. Gap analysis | not started | needs phase 1 |
| 3. Architecture proposal | not started | |
| 4. Implementation | items already under way are listed below | each on its own branch, gated before merge |
| 5. Verification | per item | `docs/testing.md`, `tools/verify.sh`, the `evals/` directory of each item |
| 6. Documentation | per item | roadmap dated entries, this page, `docs/model-file.md` |

## Work already under way that maps onto the path

| Item | Path section | Status | Evidence |
|---|---|---|---|
| Per-stage GPU profile of the dense decode window (`--profile`) | 2, dense CUDA execution | implemented | `evals/2026-11-18/halocline-dense-profile` |
| Decode attention that reads each KV cell once per window (S1 part A) | 2, attention | experimental, not shipped: bitwise, but 4% slower on 137K prose | homelab eval `halocline-s1-ab`; branch `s1-decode-attn` |
| A bandwidth-bound K8V4 decode kernel for windows of up to 8 rows | 2, attention | in development | target from the bench: one pass at 800 GB/s or more, against 474 today |
| A frequency-ranked draft head for the MTP drafter (FR-Spec, arXiv 2502.14856) | 2, speculative decoding | in development | the drafter is 15-21% of every decode window (profile above) |
| Draft depth sweep (`--spec` 3/4/6/8) | 2, speculative decoding | measured, no change | `--spec 8` best or tied at 36K and 137K |
| Quality against BF16 at depth, target-token dNLL against NInfer | 3 | measured | `evals/2026-11-18/ninfer-depth-ppl`, `evals/2026-11-17/halocline-d8-depth` |
| `tools/verify.sh`: every check that needs no GPU, in one command | 3, reproducibility | implemented on the `maturity` branch | `docs/testing.md` |

## The brief (maintainer, 2026-11-18, verbatim)

HALOCLINE_DENSE_EVOLUTION.md

Objective

Evolve Halocline's dense inference capabilities toward NInfer's level of engineering maturity while retaining Halocline's existing architecture, development direction, CUDA optimizations, and transparency standards.

This is an additive evolution, not a rewrite.

1. Audit Before Changing Anything

Thoroughly inspect the existing repository before proposing or implementing changes.

Identify:

* Existing dense inference implementation and execution paths.
* Current .halo image format, capabilities, and limitations.
* Existing model loading, tensor representations, and conversion logic.
* CUDA kernels and supported operations.
* Quantization support and quality validation.
* KV-cache implementation and memory management.
* Existing benchmarks, tests, and hardware constraints.
* Planned work already documented in RESEARCH.md, roadmap, and other project documentation.

Map every proposed improvement to existing functionality. Identify overlap, incomplete implementations, dependencies, and potential conflicts.

Do not duplicate existing work or replace functional components simply to match another project's architecture.

2. Dense Inference Development Targets

Evaluate and progressively develop the following capabilities:

Native Model Images

Evolve the existing .halo format into a comprehensive dense model artifact.

Investigate:

* Versioned binary headers and section directories.
* Tensor metadata, dimensions, layouts, and data types.
* Per-tensor quantization metadata and scales.
* Aligned tensor storage and efficient GPU loading.
* Integrity checks and per-tensor checksums.
* Model architecture and configuration metadata.
* Tokenizer and chat template packaging.
* Optional auxiliary components.
* Conversion manifests and source checkpoint provenance.

Do not introduce a competing image format if the existing format can be extended cleanly.

Conversion Pipeline

Develop a reproducible conversion pipeline for supported Hugging Face Safetensors and GGUF sources.

Prioritize:

* Native Halocline image generation.
* Tensor mapping and validation.
* Quantization conversion.
* Source-to-output integrity checks.
* Conversion reproducibility.
* Explicit handling of unsupported tensors and architectures.

Dense CUDA Execution

Evaluate the existing implementation and identify opportunities for optimized:

* GEMM and matrix multiplication.
* Attention and Flash Attention.
* Normalization and RoPE.
* MLP execution.
* Memory movement and tensor layout.
* KV-cache allocation and management.

Investigate CUDA Graphs, continuous batching, chunked prefill, and speculative decoding where compatible with the existing architecture.

Quantization

Establish a validated progression of dense weight representations:

* BF16 reference.
* FP8.
* NVFP4.
* INT8.
* Groupwise integer quantization where appropriate.

Do not add formats simply for feature completeness. Every implementation must have measurable quality and performance justification.

3. Quality and Performance Standards

Establish automated validation against BF16 reference execution.

Measure:

* KLD and perplexity.
* Top-1 agreement.
* Output correctness.
* Prefill throughput.
* Decode throughput.
* Peak VRAM usage.
* Context-length scaling.
* Conversion integrity.
* Performance consistency.

Keep measurements reproducible and disclose hardware, software versions, configurations, and limitations.

Never present compilation or mock execution as proof of GPU correctness.

4. Initial Reference Target

Use a dense Qwen 27B model as the initial end-to-end reference target.

Establish:

1. Source checkpoint validation.
2. Native .halo image generation.
3. BF16 reference execution.
4. At least one optimized quantized representation.
5. Quality comparison against BF16.
6. CUDA performance measurements.
7. Reproducible conversion and inference instructions.

Use existing Halocline hardware targets and supported architectures. Do not compromise existing sm_89 optimization priorities to pursue unrelated hardware support.

5. Competitive Evaluation

Use Basalt and NInfer as engineering references, not architectural templates.

Evaluate:

* Model image completeness.
* Conversion reliability.
* Dense execution architecture.
* Kernel efficiency.
* Memory management.
* Quantization quality.
* Runtime capabilities.
* Benchmark reproducibility.
* Documentation and external verifiability.

Identify where Halocline can match or exceed their capabilities without unnecessary complexity.

6. Non-Negotiable Project Constraints

* Preserve all existing functional implementations.
* Preserve Halocline's current development direction and Qwen4 work.
* Do not regress existing performance or quality gates.
* Do not silently change model formats or break backward compatibility.
* Do not duplicate existing functionality.
* Do not remove documented research, historical decisions, or experimental results.
* Keep changes modular and independently testable.
* Document architectural decisions and explain why alternatives were rejected.
* Maintain chronological research history and transparent engineering records.
* Distinguish implemented, experimental, planned, and unverified capabilities.

7. Execution Instructions

Work in phases:

Phase 1: Repository audit
Produce a detailed capability map and identify existing functionality that satisfies each target.

Phase 2: Gap analysis
Compare Halocline against Basalt and NInfer. Prioritize improvements by engineering value, dependencies, and compatibility.

Phase 3: Architecture proposal
Document the proposed dense evolution, including changes to existing components, new components, compatibility considerations, and migration requirements.

Phase 4: Implementation
Implement improvements incrementally. Preserve existing behavior and validate each change before proceeding.

Phase 5: Verification
Run relevant CPU and GPU tests, quality evaluations, and performance benchmarks. Record failures honestly.

Phase 6: Documentation
Update the roadmap, research history, architecture documentation, format specifications, conversion instructions, and benchmark records.

Final Deliverable

Produce a complete, technically grounded dense evolution plan and implement the highest-priority improvements that are safe to make within the existing architecture.

Clearly identify:

* What already exists.
* What was improved.
* What was added.
* What remains incomplete.
* What requires hardware validation.
* What should happen next.

The objective is not to turn Halocline into Basalt or NInfer. It is to make Halocline's dense inference stack a competitive, independently reproducible engine while preserving everything that makes the project distinct.
