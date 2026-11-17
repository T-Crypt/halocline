> [!NOTE]
> Internal design note from 2026-11-16, kept as written apart from host, path and name scrubbing. It predates the
> engine. `SPECS.md` and `RESEARCH.md` describe what was built, and the gates in `gates/` record each step.

# Basalt-Ada — SM_120 Areas, Replacement Plan & Engine Research

**Prepared:** 2026-11-16 (design session)  
**Scope:** deep research for the breakaway engine (own identity, forked from `Ex3c4Def/basalt_ada`). No implementation/builds.  
**Companion:** [`RESEARCH_PREP.md`](./RESEARCH_PREP.md), [`REVIEW_AND_STAGING.md`](./REVIEW_AND_STAGING.md), `project.md`.  
**Source of truth after 2026-11-16 audit:** [`BUILD_SPEC.md`](./BUILD_SPEC.md); corrections in [`AUDIT_LOG.md`](./AUDIT_LOG.md).

---

## 1. SM_120 areas in basalt-ada and planned replacements

### 1.1 Build system (CMake)
| Area | File | SM_120 assumption | Replacement for sm_89 |
|---|---|---|---|
| CUDA arch default | `CMakeLists.txt` L3-4 | `CMAKE_CUDA_ARCHITECTURES` defaults to `120` | Default to `89`, **sm_89 only in v1** (maintainer: native sm_89; no sm_86/multi-arch) |
| Arch guard | `CMakeLists.txt` foreach loop | Fatal if base != 120 | Remove 120-only guard; accept 89; gate sm_90+ features by arch macros |
| CUDA version floor | `CMakeLists.txt` | Requires CUDA >= 13.0 | Relax to installed the workstation toolkit; gate CUDA-13-only features |
| GPU select | `src/program/gpu_select.cpp` | Filters to sm_120 only | Filter to sm_89 (capability 8.9) only |
| Source comments | `src/models/qwen4exp/load/sources.cpp` | CUDA 13 kit note | Update to note sm_89 toolchain |

### 1.2 Cluster barriers / TMA / cp.async bulk (sm_90+)
**Files affected (9):**
- `src/models/qwen4exp/kernels/dec_gdn_rec.cu` (`__cluster_dims__`, `cg::cluster_group`, `barrier.cluster.*`)
- `src/models/qwen4exp/kernels/hc_fused.cu` (TMA descriptors, `mbarrier.init.shared::cta`, `fence.mbarrier_init.release.cluster`, `mbarrier.try_wait.parity`, `cp.async.bulk...mbarrier::complete_tx::bytes`)
- `src/models/qwen4exp/kernels/attn_chain_bound.cuh`, `attn_chain_kernel.cuh`
- `src/models/qwen4exp/kernels/dense_tc.cu`
- `src/models/qwen4exp/kernels/qsa_mid.cu`
- `src/models/qwen4exp/kernels/dec_moe.cuh`
- `src/models/qwen4exp/kernels/qsa_select.cu`
- `src/kernels/cuda/sampler.cu`
- `src/models/qwen4exp/kernels/mid_chain.cu` (TMA)

**SM_120/sm_90+ features used:**
- `barrier.cluster.arrive/wait` (cluster barriers) — sm_90+
- `fence.mbarrier_init.release.cluster` — sm_90+
- `mbarrier.try_wait.parity` (suspend form) — sm_90+
- `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` (TMA bulk copy) — sm_90+
- `__cluster_dims__` (thread block clusters) — sm_90+
- `CUtensorMap` / TMA descriptors — sm_90+

**Replacement for sm_89:**
- Remove cluster-barrier paths; use standard `__syncthreads()` / `__threadfence()` / shared-memory barriers.
- Replace TMA bulk copies with `cp.async` (sm_80+) or manual shared-memory staging with double buffering.
- Replace `mbarrier.try_wait.parity` with spin-wait or `__syncthreads()`-based synchronization.
- Replace `__cluster_dims__` kernels with standard block launches; redistribute work across blocks.
- Gate all sm_90+ paths behind `#if __CUDA_ARCH__ >= 900` or separate TUs; provide sm_89 fallback implementations.
- **Reference:** NInfer v3-port record (`reference/records/ninfer-v3-port.md`) documents exactly this pattern: "mbarrier.try_wait (suspend form) and fence.mbarrier_init.release.cluster are sm_90+ — arch-gate them; PDL device intrinsics must stay arch-gated."

### 1.3 NVFP4 / FP4 (Blackwell-only)
**Files affected (15):**
- `src/models/qwen4exp/prefill/layout.cpp`, `layout.hpp`
- `src/models/qwen4exp/prefill/fp4/moe_fp4.cu`
- `src/models/qwen4exp/prefill/experts.hpp`, `expert_fp16.cpp`
- `src/models/qwen4exp/kernels/dec_moe.cuh`
- `src/prefill/moe_mmq.cu`, `src/prefill/mmq/mmq.cuh`, `mmq_common.cuh`, `mmq_load_tiles.cuh`, `mmq_config.cuh`, `instances/mmq_nvfp4.cu`
- `src/kernels/cuda/iq_kernels.cu`
- `src/kernels/cpu/native_expert.cpp`, `nvfp4_avx512.cpp`, `nvfp4_avx2.cpp`, `nvfp4_avx2_vnni.cpp`
- `src/kernels/cpu/ggml/ggml_quants.c`, `ggml_cpu_dots.c`
- `src/program/cli/help.cpp`

**SM_120 assumption:** NVFP4 uses Blackwell FP4 tensor cores (`mxf4nvf4`, `e2m1`); basalt has extensive NVFP4 paths (layout, MoE fused, MMQ, CPU dequant).

**Replacement for sm_89:**
- **Remove NVFP4 as a first-class path on sm_89** (per NInfer decision: "NVFP4 is absent from the port, not gated" — Blackwell-only PTX).
- Replace with **fp8 (e4m3)** for weights/activations (sm_89 supports `m16n8k32 e4m3` mma).
- Use **INT8 / INT4 (Q4_K, IQ4_XS, etc.)** via existing ggml MMQ paths for MoE experts.
- Keep NVFP4 code behind `#if __CUDA_ARCH__ >= 1000` or separate build targets for future Blackwell support.
- **Reference:** NInfer v3-port: "nvfp4 sources compiled out on sm_89 + throwing stub TU (`src/ops/nvfp4_family_disabled.cpp`) covers host dispatch."

### 1.4 mma.sync tensor-core (sm_80+ but tuned for sm_120)
**Files:** 73 matches across kernels (dec_moe, dec_gdn_rec, hc_fused, dense_tc, qsa_mid, etc.)

**SM_120 assumption:** `mma.sync.aligned.m16n8k16` with specific tile shapes (16×8×16) tuned for Blackwell; `cp.async` double-buffering; Split-K scheduling.

**Replacement for sm_89:**
- sm_89 supports `mma.sync.aligned.m16n8k16` (fp16/bf16) and `m16n8k32` (fp8/int8). The instruction itself is available, but tile shapes and occupancy differ.
- Retune tile sizes for Ada (128 SMs, 24 GB VRAM); use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at runtime (per NInfer lesson).
- Replace sm_120-specific `cp.async` bulk (TMA) with standard `cp.async` (sm_80+).
- Gate sm_120-only tile shapes behind arch conditionals.

### 1.5 Cooperative launch / occupancy
**Files:** 5 matches (`gpu_select`, and kernel launch sites).

**SM_120 assumption:** Static cooperative-residency constants sized for Blackwell (1792 threads/SM, 1008 regs, etc.).

**Replacement for sm_89:**
- Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at runtime (NInfer lesson #2).
- Make workspace-capacity functions read the same occupancy value (avoid contract drift).
- Test on 4090 with 128 SMs.

### 1.6 GDN (Gated Delta Net) kernels
**Files:** `dec_gdn_rec.cu`, `dec_gdn_parity.cu`, `fused_gdn.cu`, `native_gdn.cu`, `native_gdn_preprocess.cu`, `gdn_chunked.cu`, `gdn.cu`.

**SM_120 assumption:** Cluster-based GDN recurrence (4-block clusters per value head), TMA bulk copies for state, `barrier.cluster` sync.

**Replacement for sm_89:**
- Remove cluster-based recurrence; use standard block-per-head or block-per-row launch.
- Replace TMA state transfers with `cp.async` or manual staging.
- Use NInfer's replayssm approach (GDN state checkpoints on host) as reference.
- **Reference:** TokenSpeed/FlashInfer dossiers: GDN on sm_89 falls back to Triton/FLA; FlashInfer GDN is SM90+ only. NInfer has native GDN on sm_89 (replayssm).

### 1.7 QSA (Qwen Sparse Attention) kernels
**Files:** `qsa.cu`, `qsa_mid.cu`, `qsa_select.cu`, `qsa_dec_attn.cu`, `qsa_dec_fused.cu`, `qsa_prompt_attn.cu`, `qsa_topk_parity.cpp`, etc.

**SM_120 assumption:** TMA for indexer KV, cluster barriers for top-k selection, sm_120-tuned tile sizes.

**Replacement for sm_89:**
- Use fa2 attention (sm_89 path) for QSA full-attention layers.
- Replace TMA indexer with standard global-memory loads or `cp.async`.
- Retune top-k selection for Ada occupancy.
- **Reference:** SGLang dossier: QSA on sm_89 uses baseline Triton/FLA; vLLM QSA has SM121-specific prefill kernel (#55430).

### 1.8 Gated residual / hc_fused
**Files:** `hc_fused.cu`, `hc8_gr.cu`, `fused_gr.cu`, `native_gr_norm.cu`, `native_gr_postops.cu`.

**SM_120 assumption:** TMA descriptors, mbarrier cluster barriers, `fence.mbarrier_init.release.cluster`.

**Replacement for sm_89:**
- Standard shared-memory staging with `__syncthreads()`.
- Remove cluster-barrier sync; use block-level sync.

### 1.9 MoE dispatch / dec_moe
**Files:** `dec_moe.cu`, `dec_moe.cuh`, `native_moe.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `expert_dispatch.cpp`, `expert_fp4.cpp`, `expert_fp16.cpp`, `expert_mmq.cpp`, `expert_stream.cpp`, `expert_fused.cpp`, `cpu_experts.cpp`, `shared_expert.cpp`.

**SM_120 assumption:** NVFP4 expert paths, cluster-based MoE, sm_120-tuned MMQ.

**Replacement for sm_89:**
- Remove NVFP4 expert path; use fp8/int8/Q4_K/IQ4_XS via ggml MMQ.
- Standard block launch for MoE dispatch (no clusters).
- Use NInfer's MoE offload patterns (expert residency, tiered) as reference.

### 1.10 Sampler
**File:** `src/kernels/cuda/sampler.cu` (TMA/mbarrier).

**Replacement:** Standard shared-memory staging; remove cluster barriers.

### 1.11 Vision (ViT)
**Files:** `src/models/qwen4exp/load/vision.cpp`, `serve/vision.cpp`, `src/vision/*`.

**SM_120 assumption:** Likely uses TMA for patch embedding; needs audit.

**Replacement:** Standard conv2d/linear paths; use cuDNN or manual kernels for sm_89.

---

## 2. Engine research summary (leading C++/CUDA engines for sm_89)

### 2.1 Tier 1: Our own lineage (already in lab)
| Engine | License | sm_89 status | Key value for breakaway |
|---|---|---|---|
| **NInfer-4090** (`T-Crypt/ninfer-4090`) | Apache-2.0 | Native sm_89 | Mature dense+vision+MTP serving; E8 KV; replayssm; paged KV; prefix reuse; CUDA graphs; OpenAI/Anthropic API; llama-swap integration |
| **Strata** (`T-Crypt/strata`) | MIT | Native sm_89 | MoE expert offload (VRAM+RAM); tiered residency; calibration; 4090 perf numbers; benchmark methodology |
| **basalt-ada** (`Ex3c4Def/basalt_ada`) | MIT | Blackwell only (port target) | MoE offload machinery (tiers, peer GPUs, adaptive swaps); `.basalt` format; CUDA graphs; bench/probe tools; Python serve |

### 2.2 Tier 2: Community engines (sm_89-relevant)
| Engine | License | sm_89 status | Key value |
|---|---|---|---|
| **llama.cpp** | MIT | Native (CUDA backend) | GGUF format; broad quant support (IQ4_XS, Q4_K, etc.); CPU+GPU; ik_llama.cpp fork adds MoE MMQ, QSA block scoring, Hadamard CUDA graphs |
| **ik_llama.cpp** | MIT | Native | MoE MMQ one-block-per-row; qwen4exp block scoring; fused grouped RMS norm; shared expanded MLA prefill; CUDA radix-select indexer top-k |
| **ExLlamaV3** | MIT | Native (Ada/Blackwell recommended) | Fast GPU inference; ExL2 quantization; single-user workstation focus |
| **LMDeploy (TurboMind)** | Apache-2.0 | Native (sm_89 listed) | C++/CUDA; 1.8x throughput vs vLLM; AWQ INT4/MXFP4; blocked KV cache; dynamic split&fuse; tensor parallel |
| **PowerInfer** | Apache-2.0 (Tiiny-AI) | Native (consumer GPU) | Sparse-neuron activation; GPU-CPU hybrid; hot neuron preloading; ~11x speedup on LLaMA II |
| **FastLLM** (ztxz16) | Apache-2.0 | Native (CUDA) | C++ no PyTorch dep; dense+MoE; tensor parallel; any 10GB+ GPU; DeepSeek full-precision on single GPU |
| **LightLLM** (ModelTC) | Apache-2.0 | Native (CUDA) | Python-based; token-level KV cache; efficient router scheduling; CacheTensorManager |
| **PolyStrata** (simplexai-labs) | (check) | sm_61 (Pascal) target | Tiered memory (VRAM→RAM→disk); hot/warm/cold experts; runs on 12GB cards |

### 2.3 Tier 3: Datacenter frameworks (sm_89 partial/fallback)
| Engine | License | sm_89 status | Key value |
|---|---|---|---|
| **vLLM** | Apache-2.0 | Partial (GDN Triton/FLA fallback; fast paths SM90+) | PagedAttention; continuous batching; prefix caching; MTP; vision; 256K+ |
| **SGLang** | Apache-2.0 | Partial (GDN Triton; fast paths SM90+) | RadixAttention; Unified Radix Cache; MTP/EAGLE/DSpark; HiCache; PD disaggregation |
| **TensorRT-LLM** | Apache-2.0 | Partial (GDN Triton; fast paths SM90+) | NVIDIA kernels; FP8 on Ada; MTP for Qwen3.5 MoE; CUDA graphs |
| **FlashInfer** | Apache-2.0 | Partial (fa2 on sm_89; GDN SM90+) | Kernel library; attention/GEMM/MoE/sampling; fa2 on Ada; cuDNN BF16 GEMM on sm_89 |
| **KTransformers** | Apache-2.0 | MoE only (dense not supported) | Expert offload (GPU→CPU→disk); 256K context; prefix cache (kvc2) |
| **TokenSpeed** (LightSeek) | MIT | Unverified (Triton/FLA fallback) | Control/execution plane separation; FlatKV; MTP/DSpark; GDN native |

### 2.4 Tier 4: Chinese sources / repos
| Source | Type | Relevance |
|---|---|---|
| **FastLLM** (ztxz16/fastllm) | C++ engine, no PyTorch dep | Dense+MoE; tensor parallel; expert cache docs; context extension docs; hybrid inference docs |
| **LMDeploy** (InternLM/lmdeploy) | C++/CUDA (TurboMind) | sm_89 explicitly supported; AWQ INT4; 1.8x throughput; EuroSys 2027 paper |
| **PowerInfer** (Tiiny-AI) | C++ engine | Sparse neuron activation; GPU-CPU hybrid; hot neuron preloading |
| **LightLLM** (ModelTC) | Python serving | Token-level KV cache; efficient router; research-friendly |
| **marlin kernel** | CUTLASS-based | 4-bit weight quantization; Ampere/Ada tensor cores; cp.async; used in vLLM |
| **Humming** (Ant Group) | Kernel library | General-purpose quantized kernels; open-sourced via vLLM |

### 2.5 PolyStrata (simplexai-labs) — verified 2026-11-16
- **Repo:** <https://github.com/simplexai-labs/PolyStrata>, MIT, built on Strata (MIT). Qwen engine/installer/server/web app are Strata's; PolyStrata adds a GLM 5.3 Flash engine. Parts under `third_party/ggml` (MIT); one experimental data vector under Qwen Community License 1.0; model files carry their own licenses.
- **What:** Tiered memory model. GPU (VRAM): attention + DeltaNet mixers, gated-residual weights, routers, shared experts, output head, MTP draft layer, KV cache (from 64K only the hot part resident, rest streams from RAM), plus an adaptive expert cache (~700 more experts per extra GB). RAM: every expert pinned; CPU computes non-resident experts in place with AVX-512/AVX2 (ggml i-quants) concurrently with GPU. SSD: ~28.8 GB n-gram table read a few rows per token via OS cache.
- **Target:** same engine compiled for NVIDIA (CUDA) and AMD (HIP). `strata-poly` engine for GLM targets sm_61 with CUDA 12.8 (community report, not primary); the project is not sm_89-first.
- **Models:** Qwen3.8-Flash-Next (512 experts, 10 active), GLM 5.3 Flash (24,576 experts, 10 active), DeepSeek V4 Flash, LFM2.5, Qwen3.6. The "24,576 experts" figure belongs to GLM, not Qwen.
- **Spec decode:** in-model MTP drafts up to 3 tokens, verified over 48 layers at 2.4-3.2 tokens/pass (1.6-1.8x); prompt lookup drafts up to 5 tokens on repeated context (code edits 6-11% faster); drafts are verified so output is unchanged.
- **Value for our project:** concrete production pattern for VRAM expert cache + pinned RAM CPU compute + SSD n-gram tier, and for KV streaming beyond 64K. Confirms the tiering concepts in basalt `tiers.cpp`/Strata. Concepts only; our engine stays native sm_89 CUDA C++/CUDA.

### 2.6 Key papers (from dossiers + web search)
| Paper | Venue | Relevance |
|---|---|---|
| **PowerInfer** (arXiv 2312.12456) | — | Sparse neuron activation; GPU-CPU hybrid inference on consumer GPUs |
| **LMDeploy** (arXiv 2508.15601) | EuroSys 2027 | Mixed-precision inference; TurboMind; 61% lower latency, 156% higher throughput |
| **DFlash** (arXiv 2602.06036) | ICML 2026 | Block diffusion speculative decoding; 7.7 tok/round on Qwen3.5-27B |
| **FlashInfer** (arXiv 2501.01005) | — | Kernel library; fa2 on sm_89; GDN SM90+ only |
| **SGLang** (arXiv 2312.07104) | NeurIPS 2024 | RadixAttention; 6.4x throughput on agent workloads |
| **PagedAttention** (SOSP 2023) | SOSP | KV cache paging; near-zero waste |
| **Qwen3.8-Flash-Next tech report** | QwenLM GitHub | QSA sparse attention; gated residual; PLE n-gram embedding; 262K native |
| **KTransformers** (SOSP paper) | SOSP | MoE expert offload; 3-tier memory (GPU→CPU→disk) |
| **VisionPsy-Nano** (arXiv 2609.31746) | — | IQ4_XS quality retention (<1 point from FP32) |
| **Which Decisions Low-Bit Quantization Breaks** (arXiv 2608.06564) | — | Decision-level quality analysis of IQ4_XS vs Q4_K_M |
| **Bit-Exact AI Inference Verification** (arXiv 2606.00279) | — | Verification without performance tradeoffs; SM89 CUTLASS tactics |

---

## 3. Replacement strategy summary

### 3.1 Immediate (M1 — build skeleton)
1. CMake: default arch 89, remove 120-only guard, relax CUDA 13 floor.
2. gpu_select: filter to sm_89.
3. Guard all sm_90+ kernel paths behind `__CUDA_ARCH__ >= 900` macros.
4. Provide sm_89 fallback (standard barriers, cp.async, manual staging).

### 3.2 Short-term (M2 — kernel foundation)
1. **GDN recurrence:** remove cluster-based; standard block launch + replayssm (NInfer reference).
2. **QSA attention:** fa2 path for full-attention layers; standard loads for indexer.
3. **MoE dispatch:** standard block launch; remove NVFP4; use fp8/int8/Q4_K via ggml MMQ.
4. **Gated residual/hc_fused:** standard shared-memory staging.
5. **Sampler:** standard staging.
6. **Occupancy:** runtime `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

### 3.3 Medium-term (M3-M5)
1. **Dense adapter:** add Qwen3.8-27B dense (reuse GDN/attention/ViT from qwen4exp).
2. **E8 KV:** port rk4v4-e8/rk2v4-e8 for context extension (98.7% / 96.2% cosine vs FP32; INT8 is 99.8% but caps context at ~223K).
3. **Speculative decoding:** MTP (lossless) + DFlash (block diffusion, +50% throughput).
4. **Quality gates:** imatrix-aware IQ4_XS conversion, perplexity/NIAH/task-accuracy automation.
5. **Tiered offload:** VRAM + pinned RAM + disk (Strata/basalt/PolyStrata patterns).

### 3.4 What to borrow from each engine (concepts, not code)
| Concept | Source engine | sm_89 applicability |
|---|---|---|
| E8 lattice KV quantization | UDPSendToFailed/ninfer-4090 | High (proven on 4090) |
| ReplaySSM (GDN state checkpoints) | NInfer | High (native sm_89) |
| Persisting-L2 pinning for drafter weights | UDPSendToFailed | High (Ada-specific) |
| Ada tensor-core small-T decode path | UDPSendToFailed | High (mma.sync.m16n8k16 available on sm_89) |
| Expert offload (VRAM→RAM→disk) | Strata, basalt, PolyStrata, KTransformers | High |
| Adaptive expert usage seeding | basalt (tiers.cpp) | High |
| Paged KV cache | vLLM, SGLang, NInfer | High |
| RadixAttention (prefix reuse) | SGLang | Medium (needs sm_89 kernels) |
| Blocked KV cache + dynamic split&fuse | LMDeploy (TurboMind) | Medium |
| Sparse neuron activation | PowerInfer | Medium (different from MoE expert offload) |
| Token-level KV cache | LightLLM | Low (Python-based) |
| Tiered memory (VRAM→RAM→disk) | PolyStrata | High (concept) |

---

## 4. What we are NOT doing
- **Not building "basalt"** — new engine identity from the unique-name bucket.
- **Not targeting Blackwell** — sm_89 first; Blackwell support can be added later via arch guards.
- **Not copying code blindly** — license/provenance audit before any code reuse; prefer clean-room reimplementation for kernel hot paths.
- **Not building yet** — this is research + planning only.

---

## 5. Open questions (updated)
1. **Name:** pick 3-5 from unique-name bucket for verification?
2. **Base commit:** pin `Ex3c4Def/basalt_ada` commit for fork?
3. **First dense model:** Qwen3.8-27B dense (reuse qwen4exp GDN/attention/ViT) or smaller?
4. **First sparse model:** Qwen3.8-Flash-Next IQ4_XS (Strata baseline for comparison)?
5. **Artifact format:** `.basalt` (native) vs bridge to `.ninfer`/GGUF?
6. **KV quality target:** rk4v4-e8 (98.7% cosine) or rk2v4-e8 (96.2% cosine) for context extension; INT8 (99.8%) is the high-fidelity reference but caps context?
7. **Host RAM assumption:** 31 GB (Strata-like) or 64 GB?

*End of research + planning document. No implementation.*  
