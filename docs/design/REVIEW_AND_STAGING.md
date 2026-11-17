> [!NOTE]
> Internal design note from 2026-11-16, kept as written apart from host, path and name scrubbing. It predates the
> engine. `SPECS.md` and `RESEARCH.md` describe what was built, and the gates in `gates/` record each step.

# Basalt-Ada Engine — Review & Staging Plan

**Prepared:** 2026-11-16 (design session)  
**Scope:** review of the basalt-ada-lift prep work, name collisions, core code, speculations, and a staged plan (no implementation/builds).  
**Companion:** [`RESEARCH_PREP.md`](./RESEARCH_PREP.md), `project.md`.

---

## 1. Name collisions review (deep web search)

### 1.1 "Basalt" — already heavily used
- **Basalt VIO** (visual-inertial odometry library, OpenSLAM-org) — widely referenced.
- **Basalt** database / MCP server (`yoshiii-xyz/basalt`).
- **basalt_ros2** — ROS2 wrapper.
- **basalt** TUI app for managing notes (`erikjuhani/basalt`).
- **basalt-launcher** — Minecraft launcher.
- **basalt** inference engine (`jesdga95/basalt`, `Ex3c4Def/basalt_ada`) — the upstream we fork from.
- **Conclusion:** "Basalt" is **not unique** and will collide with VIO, DB, TUI, game launcher, and upstream engine naming. Using it for a breakaway engine risks confusion, SEO noise, and trademark/branding friction.

### 1.2 "Basalt Ada" / "basalt-ada"
- "Ada" is the name of a programming language (Ada/SPARK) and a common project name.
- No direct "Basalt Ada" inference-engine collision found, but the combination inherits all "Basalt" collisions and adds "Ada" ambiguity (language vs GPU arch).
- **Conclusion:** avoid "Basalt Ada" as a final identity; keep it as a working label only.

### 1.3 Other common engine names checked (collisions found)
| Candidate | Collision evidence |
|---|---|
| Granite | IBM Granite models; granite-switch; granitepi; granite inference references |
| Obsidian | Obsidian app ecosystem; Obsidian Memory Transformer (LLM) |
| Ember | `hammy229/ember` (LLM engine from scratch, C++20); Project Ember (Mithril, compound AI); Muse Ember (Meta) |
| Slate | `slate-engine` (local AI engine for macOS); Slate workspace |
| Onyx | `onyx-dot-app/onyx` (open source AI platform, 33k stars) |
| Quartz | `SaientAI/saient-quartz` (on-device inference engine, Rust) |
| Garnet | Garnet profile-based AI inference runtime/compiler |
| Topaz | Topaz Labs upscaling; Topaz image API |
| Opal | IBM opal-sim (LLM platform simulator); Opal private memory |
| Marble | MARBLE multi-agent reasoning engine (LLM) |
| Hematite | hematite-cli (local AI workstation) |
| Pyrite | pyrite-stream (MoE-Stream Engine) |
| Malachite | Malachite consensus engine (BFT, Rust) |
| Peridot | peridot-local (sovereign local kernel) |
| Spinel | Spinel Ruby compiler |
| Fluorite | Mithril Security "fluorite" (high-throughput inference engine) |
| Basalt (upstream) | basalt VIO, basalt DB, basalt TUI, basalt Minecraft launcher |

### 1.4 Unique-name bucket (no direct inference-engine collision found in searches)
These are mineral/rock-derived or coined names with **no LLM inference-engine project found** in web/GitHub searches. Still require final verification (GitHub name availability, PyPI/crates.io, trademark) before committing.

| Name | Origin/feel | Notes |
|---|---|---|
| **Tufa** | Porous limestone formed by mineral springs | Short, pronounceable, no collision found |
| **Gneiss** | High-grade metamorphic rock (banded) | Distinctive; "gn" spelling may aid uniqueness |
| **Schist** | Medium-grade metamorphic rock | Memorable; no inference-engine hit |
| **Pumice** | Volcanic rock (light, porous) | Evokes lightweight/efficient; no collision |
| **Zircon** | Gemstone/mineral | Strong, short; no inference-engine hit (only mineral-exploration ML) |
| **Apatite** | Phosphate mineral | No inference-engine hit |
| **Calcite** | Common carbonate mineral | No inference-engine hit |
| **Jade** | Gemstone | No inference-engine hit (person names only) |
| **Agate** | Banded chalcedony | No inference-engine hit |
| **Jasper** | Opaque chalcedony | No inference-engine hit (person names only) |
| **Azurite** | Blue copper mineral | No inference-engine hit |
| **Turquoise** | Blue-green mineral | No inference-engine hit |
| **Lapis** | Lapis lazuli (blue stone) | No inference-engine hit (only "Lapis Occultus" in vitriol) |
| **Tourmaline** | Borosilicate mineral | No inference-engine hit |
| **Beryl** | Beryllium mineral (emerald/aquamarine family) | No inference-engine hit |
| **Amethyst** | Purple quartz | No inference-engine hit |
| **Citrine** | Yellow quartz | No inference-engine hit |
| **Cinder** | Volcanic fragment | No inference-engine hit (Cinder CSI driver, game refs only) |
| **Corundum** | Aluminum oxide (ruby/sapphire) | Only hit: Corundum ethernet-switch (not inference) |
| **Tephra** | Fragmental volcanic material | No hit |
| **Scoria** | Vesicular volcanic rock | No hit |
| **Tuff** | Consolidated volcanic ash | No hit |
| **Ignimbrite** | Pyroclastic flow deposit | No hit |
| **Lahar** | Volcanic mudflow | No hit |
| **Breccia** | Angular rock fragments | No hit |
| **Pegmatite** | Coarse-grained igneous rock | No hit |
| **Skarn** | Metasomatic rock | No hit |
| **Greisen** | Hydrothermal alteration rock | No hit |
| **Diorite** | Igneous rock | No hit |
| **Gabbro** | Mafic igneous rock | No hit |
| **Peridotite** | Ultramafic rock | No hit |
| **Eclogite** | High-pressure metamorphic rock | No hit |
| **Migmatite** | Mixed metamorphic/igneous rock | No hit |
| **Phyllite** | Low-grade metamorphic rock | No hit |
| **Trachyte** | Volcanic rock | No hit |
| **Andesite** | Intermediate volcanic rock | No hit |
| **Rhyolite** | Felsic volcanic rock | No hit |
| **Dacite** | Intermediate volcanic rock | No hit |
| **Syenite** | Alkaline igneous rock | No hit |
| **Monzonite** | Intermediate igneous rock | No hit |
| **Anorthosite** | Plagioclase-rich rock | No hit |
| **Kimberlite** | Diamond-bearing rock | No hit |
| **Lamproite** | Potassic volcanic rock | No hit |

**Recommendation:** pick from the short, pronounceable subset first: **Tufa, Zircon, Jade, Agate, Jasper, Lapis, Beryl, Citrine, Cinder, Scoria, Tuff, Skarn, Gabbro, Diorite, Rhyolite, Trachyte**. Avoid names with known inference-engine collisions (Granite, Obsidian, Ember, Slate, Onyx, Quartz, Garnet, Topaz, Opal, Marble, Hematite, Pyrite, Malachite, Peridot, Spinel, Fluorite, Basalt).

**Next step (not done):** verify GitHub org/repo name availability, PyPI/crates.io, and trademark for top 3–5 picks before finalizing. This is reversible planning only.

---

## 2. Core code work review (basalt-ada-lift + upstream)

### 2.1 What basalt actually is (from code, not just docs)
- **Language/stack:** C++20 + CUDA, CMake 3.24+, GCC/MSVC, Python `serve/` (OpenAI+Anthropic), ggml-derived quant/MMQ pieces in `src/` and `include/basalt/ggml/`.
- **Binaries:** `basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision` (per CMakeLists and docs).
- **Models:** only two adapters registered: `qwen4exp` (Qwen3.8-Flash-Next / Qwen4 preview, MoE) and `deepseek41` (DeepSeek-V4.1-Flash, MoE). Both are MoE architectures; **no dense model adapter exists** in this codebase. This is a significant gap for the "dense + sparse" goal.
- **Subsystems:** artifact/ggml dequant, tokenizer/grammar, serve (Python), tiers/offload (expert residency), CUDA graphs, bench/probe, vision, n-gram (speculative), MTP, GDN (gated delta net), QSA (sparse attention), gated residual, PLE (n-gram embedding).
- **Tests:** extensive parity tests (kernels, prefill, routes, MMQ, GDN, QSA, etc.) — mostly GPU tests. CPU tests exist for platform and some model logic.

### 2.2 Blackwell/sm_120 assumptions (verified in code)
- **CMakeLists.txt:** defaults `CMAKE_CUDA_ARCHITECTURES` to `120`; hard-fails unless arch base is 120; requires CUDA >= 13.0. Message: "basalt builds for compute capability 12.0 only".
- **gpu_select.cpp:** filters GPUs to sm_120 only ("no sm_120 GPU visible").
- **Kernels:** heavy use of sm_90+ features:
  - `barrier.cluster.arrive/wait`, `fence.mbarrier_init.release.cluster`, `mbarrier.try_wait.parity`, `cp.async.bulk...mbarrier::complete_tx::bytes` (cluster barriers, TMA, cp.async bulk) in `hc_fused.cu`, `dec_gdn_rec.cu`, etc.
  - `__cluster_dims__`, `cg::cluster_group`, `cg::this_cluster()`.
  - `mma.sync.aligned.m16n8k16` (tensor core) in various kernels.
  - NVFP4 paths (`GGML_TYPE_NVFP4`, `moe_fp4.cu`, `expert_fp4.cpp`) — Blackwell FP4 tensor cores.
  - XQA/FlashAttention-3-style paths likely (not fully audited).
- **Conclusion:** porting to sm_89 is not a retarget; it's a substantial kernel/architecture port comparable to NInfer v3 port (which took months). Expect to: remove/disable cluster-barrier/TMA/cp.async-bulk paths behind arch guards, replace NVFP4 with fp8/int8 paths, replace XQA with fa2, revalidate attention/MoE hot paths, and re-tune occupancy/cooperative launch for Ada.

### 2.3 Tiers / offload (expert residency)
- **Single tiers file:** `src/models/qwen4exp/serve/tiers.cpp` (628 lines) contains:
  - `seed_usage_from_prompt` — seeds adaptive tier usage from prompt routing.
  - `resident_spare_swaps` — swap logic with spare slots, copy-back, landing.
  - `QfnRuntime::adapt_peers` — multi-tier (peer GPU) adaptation.
  - `QfnRuntime::apply_pending` — pending swap application with events.
- **Implied architecture:** VRAM-resident experts + host (RAM) experts + optional peer GPU tiers; adaptive swaps based on usage; file-backed expert source (`FileExpertSource`) with spare slots; peer experts (`PeerExperts`).
- **Quality of implementation:** sophisticated, with adaptive usage tracking, peer tiers, spare-slot optimization, and CUDA event synchronization. This is a real offload engine, not a stub.
- **Risk for Ada:** the swap kernels and memory copies likely assume Blackwell memory bandwidth/pinned memory behavior; need to validate on 4090 with host RAM pressure (Strata lesson: ~21 GiB pinned RAM can OOM the box).

### 2.4 Models / adapters
- **qwen4exp:** large (load, prefill, decode, serve, kernels, MTP, GDN, QSA, gated residual, PLE, vision, conversation, grammar, sampling). This is the Qwen3.8-Flash-Next / Qwen4 preview adapter (MoE).
- **deepseek41:** DeepSeek-V4.1-Flash adapter (MoE, DSpark drafting).
- **No dense adapter.** The project goal (dense + sparse) requires either:
  - Adding a dense model adapter (e.g., Qwen3.8-27B dense, which shares architecture with Flash-Next minus MoE).
  - Or treating "dense" as a degenerate MoE with 1 expert (not recommended; overhead).
- **Recommendation:** plan a dense adapter early (M3) using Qwen3.8-27B dense as the target, leveraging the shared GDN/attention/ViT code.

### 2.5 Serving (Python `serve/`)
- OpenAI + Anthropic API, streaming, metrics, chunked body, lifecycle, security, tool params, structured output, grammar, vision, sleep, billing stamp, telemetry.
- **Quality:** production-grade serving layer with extensive tests (20+ test files).
- **Reuse:** can likely be reused/adapted with minimal changes (it's Python, arch-agnostic).

### 2.6 Tests
- Extensive parity tests (kernels, prefill, routes, MMQ, GDN, QSA, etc.) — mostly GPU.
- CPU tests for platform and some model logic.
- **Gap:** no sm_89-specific tests; no dense-model tests; no 4090-specific perf regression tests.

### 2.7 Build / CI
- GitHub Actions: `ci.yml` (builds Linux/Windows, runs CPU tests), `pages.yml` (docs).
- Upstream has additional workflows: `community.yml`, `release.yml`, `actions/` dir.
- **Gap:** no GPU tests in CI (requires card); no sm_89 build matrix.

---

## 3. Speculations review (unverified claims flagged)

| # | Claim (source) | Status | Action |
|---|---|---|---|
| S1 | "basalt_ada is behind upstream" (RESEARCH_PREP §3.2) | Shallow-clone diff (601 paths incl. `.git`); meaningful diffs verified but not exhaustively categorized | Run `git diff --stat` on full clones; pin specific commits |
| S2 | "No sm_89/4090/Ada architecture enablement found" (RESEARCH_PREP §3.1, §2.2) | Grep on clones; 4 sm_120 refs in source (gpu_select + sources.cpp), 389 NVFP4/fp4, 180 TMA/mbarrier/cluster. Kernel audit not exhaustive | Run comprehensive `rg` inventory and save as a note; verify no hidden `#ifdef __CUDA_ARCH__ >= 900` paths |
| S3 | "Recommended initial targets: dense Qwen3.8-27B, sparse Qwen3.8-Flash-Next" (RESEARCH_PREP §8) | Recommendation, not verified by code/artifact availability | Ratify with maintainer; verify local artifact paths/hashes |
| S4 | "Vault is sparse for this topic" (RESEARCH_PREP §6) | Verified by SSH inventory (datasets/mcp/rag/research empty; only generic inbox match) | Re-run if maintainer adds material |
| S5 | "Strata operational profile: ~15-16 GB VRAM + ~21 GiB pinned RAM" (project.md, RESEARCH_PREP §5) | From `guide/strata-and-ninfer.md` and releases; not live-measured in this session | Verify against `~/strata` configs/logs if needed |
| S6 | "NInfer-4090 proven choices" (RESEARCH_PREP §4) | From README/records; not independently measured here | Accept as documented; re-measure when implementing |
| S7 | "License/provenance memo" (RESEARCH_PREP §7) | Prep-level; not legal sign-off | File-level audit before code reuse |
| S8 | "basalt codebase is Blackwell/SM120-oriented (kernels, pack/quant assumptions, CUDA 13 features)" (RESEARCH_PREP §3.1) | Build gate verified; kernel assumptions partially verified (389 NVFP4, 180 TMA/cluster) | Full kernel audit during M1/M2 |
| S9 | "Dense+sparse unified runtime abstraction is the core design risk" (project.md) | Reasonable inference from code structure (two MoE adapters, no dense adapter, tiers) | Validate with prototype |
| S10 | "24 GB VRAM ceiling forces quant/KV/context/offload tradeoffs" (project.md) | From NInfer/Strata numbers; not measured for basalt | Measure during M2/M3 |
| S11 | "Format choice: .basalt vs .ninfer/GGUF affects converter effort" (project.md) | Reasonable; not quantified | Benchmark converter cost in M0 |
| S12 | "License clarity: blocker for direct code reuse from NInfer/Strata" (project.md) | Prep-level; needs file-level audit | Do audit before M1 |
| S13 | "NInfer prep4qwen dossiers contain sm_89-relevant data" (user prompt) | Verified: 13 research dossiers exist on prep4qwen branch; read 10+; contain sm_89-relevant findings (XQA not on sm_89, GDN FlashInfer SM90+ only, TRT-LLM GDN Triton fallback, etc.) | Cross-reference with M1/M2 decisions |

**Key speculations to resolve before coding:**
1. Whether basalt's tiers/offload can run on sm_89 without kernel rewrites (unlikely; cluster barriers/TMA/cp.async-bulk are sm_90+).
2. Whether a dense adapter can be added by reusing qwen4exp's GDN/attention/ViT code (likely yes, but needs verification).
3. Whether `.basalt` format can encode dense+MoE topology cleanly (needs format audit).
4. Whether host RAM on the workstation (~31 GB) can support Strata-like pinned RAM alongside a 4090 build process (historically risky).

---

## 4. Improvements over Strata and basalt (testing) while retaining high KLE / higher quality on IQ4_XS

### 4.1 Context: our main model is IQ4_XS (Strata IQ4_XS line)
- Strata runs Qwen3.8-Flash-Next IQ4_XS (expert-offload to RAM) at 262K context, ~15-16 GB VRAM + ~21 GiB pinned RAM, 139-154 tok/s decode.
- Goal: retain or improve quality (KLE / perplexity / task accuracy) at IQ4_XS while improving throughput, context, or stability.

### 4.2 Where improvement is plausible (from research + code)

#### A. Speculative decoding (MTP / DFlash / DSpark)
- **MTP (in-checkpoint):** basalt supports MTP on by default; natpate/ninfer-windows benchmarks show MTP3 gives ~2x decode speedup on Qwen3.8-27B (89-91% acceptance on structured output). Quality is mathematically lossless (speculative decoding preserves target distribution).
- **DFlash (block diffusion drafter):** arXiv 2602.06036 shows DFlash achieves higher throughput than MTP on Qwen3.5-27B (7.7 tok/round vs 5.5 for MTP) with strong speedups at low/medium concurrency. **Quality is preserved** (speculative decoding is lossless). DFlash is supported in vLLM/SGLang and natpate (window 1-15).
- **DSpark (trained GDN drafter):** TokenSpeed/SGLang support DSpark for Qwen3.8-27B; DSpark is a trained drafter (not lossless by default but can be tuned). For quality-critical work, MTP/DFlash (lossless) preferred.
- **N-gram prompt-lookup drafting:** UDPSendToFailed fork adds hierarchical n-gram drafting (N=5→4→3→2) layered alongside MTP, zero-allocation, ~12-15% prefill improvement. Quality-preserving.
- **Improvement lever:** Add DFlash support to basalt (if not present) and tune MTP depth; use lossless speculative decoding to improve decode speed without quality loss. This is the highest-ROI quality-neutral improvement.

#### B. KV cache quality/efficiency
- **E8 lattice KV (rk4v4-e8, rk2v4-e8):** UDPSendToFailed fork achieves 223K→567K context on 24 GB with 96-99% cosine vs FP32 and 100% NIAH recall. Quality retention (verified against UDPSendToFailed README, 2026-11-16): **int8 99.8%**, rk8v4 99.4%, **rk4v4-e8 98.7%**, rk4v4 97.8%, **rk2v4-e8 96.2%** cosine vs FP32. For IQ4_XS weights, using E8 KV (instead of INT8) could extend context without quality loss beyond the weight quantization.
- **YaRN 1M scaling:** UDPSendToFailed adds YaRN 1M frequency scaling + attention temperature scaling. Could extend context beyond 262K for long-context tasks.
- **Direct L1-cached block-table lookups:** lowers address overhead for 1M-token envelopes.
- **Improvement lever:** Implement or port E8 KV quantization + YaRN scaling to extend context at IQ4_XS quality. Quality gate: NIAH recall, perplexity, task accuracy.

#### C. Expert offload / tiering (sparse path)
- **Current Strata:** VRAM + pinned RAM, adaptive swaps, no disk/peer tiers in production.
- **Basalt tiers:** supports peer GPU tiers, spare-slot swaps, adaptive usage seeding. More sophisticated than Strata's.
- **UDPSendToFailed:** adds direct L1 block-table lookups, 72 MB persisting-L2 pinning for MTP proposal weights (Ada-specific), stream fork-join co-scheduling.
- **KTransformers:** GPU→CPU→disk 3-tier prefix cache (kvc2) for long-context KV reuse; but only MoE, not dense.
- **Improvement lever:** Add disk-tier expert offload (for very large MoE models) and persisting-L2 pinning for drafter weights. Validate residency/eviction under memory pressure. Quality should be unchanged (offload doesn't affect math, only latency).

#### D. Prefill optimization
- **Ada Tensor-Core small-T decode path:** UDPSendToFailed runs Q4/Q5/W8 projections on tensor cores (mma.sync.m16n8k16) with cp.async double-buffering, Split-K=2 scheduling. This is Ada-specific and could improve prefill/decode throughput.
- **GDN input-proj + causal-conv in-CTA fusion:** one kernel combining tensor-core projection → smem staging → 1D conv → state publish. Reduces kernel launch overhead.
- **Prefill chunking:** NInfer uses 1024-token chunks; tune for 4090.
- **Improvement lever:** Port or implement Ada tensor-core projection paths for Q4/Q5/Q8 small-T; fuse GDN projection+conv. Quality-neutral (same math, faster execution).

#### E. Quality retention at IQ4_XS
- **Imatrix (importance matrix):** IQ4_XS with imatrix (as in unsloth builds) retains near-Q4_K_M quality. Ensure conversion uses imatrix.
- **Layer-specific quantization:** ResearchGate paper suggests layer-specific strategies (IQ4_XS for attention, IQ2_XXS for FFN) reduce error. Could be applied to `.basalt`/`.ninfer` artifacts.
- **Perplexity gating:** basalt has `ppl_gate.py` and quality gate infrastructure; use to validate IQ4_XS quality vs FP16/BF16 reference.
- **KV quality:** Use E8 KV (rk4v4-e8) which has 98.7% cosine vs FP32 (vs INT8 99.8%), trading a small measured cosine difference for far larger resident context.
- **Improvement lever:** Add imatrix-aware conversion, layer-specific quant profiles, and automated perplexity/quality gates per model at IQ4_XS. This directly addresses "retain highly KLE / higher quality on IQ4_XS."

#### F. Context extension
- **Current:** Strata 262K at IQ4_XS; NInfer E8 KV extends to 433K (rk4v4-e8) / 567K (rk2v4-e8).
- **YaRN 1M:** UDPSendToFailed demonstrates 359K NIAH at 100% recall with rk2v4-e8.
- **Improvement lever:** Extend context beyond 262K using E8 KV + YaRN, with quality gates (NIAH, perplexity, task accuracy). This is a quality-neutral capability improvement.

#### G. Testing / benchmarking improvements
- **Strata baseline:** 139-154 tok/s decode at 262K, 139-154 tok/s at 512K (tight).
- **Basalt (Blackwell):** 239-479 tok/s decode (NVFP4, 5090), 7,317 tok/s prefill at 64K (claimed).
- **UDPSendToFailed (4090):** 272 tok/s decode MTP7 at 32K, 2,146 tok/s prefill at 2K, 100% NIAH at 359K.
- **Improvement lever:** Establish a reproducible benchmark matrix on 4090 (prefill/decode/TTFT at fixed contexts, VRAM/RAM, NIAH, perplexity) comparing against Strata IQ4_XS baseline. Use basalt-bench/probe infrastructure. Track regressions.

### 4.3 Summary: highest-ROI improvements (quality-neutral or quality-positive)
1. **Lossless speculative decoding (MTP/DFlash):** +50-100% decode speed, zero quality loss. Add DFlash if missing.
2. **E8 KV quantization + YaRN:** +50-100% context extension, minimal quality loss (98.7% cosine for rk4v4-e8; 96.2% for rk2v4-e8, vs 99.8% for INT8).
3. **Imatrix-aware, layer-specific IQ4_XS conversion:** better quality at same size.
4. **Ada tensor-core projection paths:** +20-40% prefill/decode throughput, quality-neutral.
5. **Persisting-L2 pinning for drafter weights:** reduce drafter memory traffic, quality-neutral.
6. **Automated quality gates (perplexity, NIAH, task accuracy):** ensure IQ4_XS quality retention is measured, not assumed.
7. **Disk-tier expert offload:** enable larger MoE models, quality-neutral.

---

## 5. Staging plan (no implementation)

### Stage 0 — Decisions (maintainer/architect)
- [ ] Ratify engine name from unique-name bucket (top 3-5 verified for GitHub/PyPI/trademark).
- [ ] Ratify fork base: **`Ex3c4Def/basalt_ada`** (maintainer decision: fork and port directly from basalt-ada; upstream basalt improvements cherry-picked later if needed).
- [ ] Ratify first dense + sparse model pair and artifact format.
- [ ] Ratify minimum feature set for v0 (API, KV modes, quants, max context, vision, tool-calling).
- [ ] Ratify host RAM assumption (31 GB vs 64 GB).

### Stage 1 — Discovery closure (docs/repo, no builds)
- [ ] S1.1 Fork decision record: fork from `Ex3c4Def/basalt_ada` at pinned commit; document rationale (inherit fork lineage, port directly, cherry-pick upstream later if needed).
- [ ] S1.2 License/provenance audit memo (file-level, including third-party notices).
- [ ] S1.3 Target matrix doc (models, formats, features, non-goals).
- [ ] S1.4 Deep source survey: map basalt modules and identify sm_120-specific sites (save `rg` inventory as a note).
- [ ] S1.5 Cross-reference ninfer prep4qwen dossiers for sm_89-relevant findings (XQA not on sm_89, GDN FlashInfer SM90+, TRT-LLM GDN Triton fallback, etc.) and fold into M1/M2 decisions.

### Stage 2 — sm_89 build skeleton (first code work)
- [ ] S2.1 CMake/toolchain: configurable arch (default 89), remove hard-fail, relax CUDA 13-only where incompatible.
- [ ] S2.2 Compile guards: disable/sm_90+ paths behind arch macros; minimal GPU smoke target.
- [ ] S2.3 CPU tests buildable where possible; document build commands.
- **Gate:** configure+build on the workstation; `ctest -L cpu` passes; GPU smoke runs alone on 4090.

### Stage 3 — Ada kernel & memory foundation
- [ ] S3.1 Attention/KV correctness (prefill/decode) for sm_89-supported dtypes.
- [ ] S3.2 GEMM/GEMV/MMQ/quant paths: validate or replace sm_120-tuned paths; document unsupported quants.
- [ ] S3.3 Memory management: paged KV, arena sizing, prefill chunk limits, graph capture constraints, OOM guards.
- **Gate:** kernel parity tests on 4090; greedy/logit spot checks; VRAM/RAM accounting logs.

### Stage 4 — Dense end-to-end
- [ ] S4.1 Loader/weights path for chosen dense artifact.
- [ ] S4.2 Dense runtime (forward, sampling, context/KV, no MoE overhead).
- [ ] S4.3 Minimal serve/CLI (OpenAI-compatible, streaming, health/metrics).
- **Gate:** coherent generation, stable greedy, TTFT/decode/prefill baseline recorded.

### Stage 5 — Sparse/MoE end-to-end
- [ ] S5.1 Router + expert execution on sm_89.
- [ ] S5.2 Tiered expert residency/offload (VRAM + pinned RAM + optional disk/peer).
- [ ] S5.3 Dense+sparse unification (shared scheduler/KV/serving, mode-specific hot paths).
- **Gate:** correctness on Flash-Next (or chosen MoE); throughput/latency vs Strata baseline; residency/eviction under pressure.

### Stage 6 — Productization & migration
- [ ] S6.1 Serving/ops (load/unload, metrics, config UX, llama-swap/Berth integration).
- [ ] S6.2 Validation matrix (parity gates, long-context/tool-call/grammar/vision, perf regression tracking).
- [ ] S6.3 Docs & breakaway (engine name, build/run docs, supported configs, migration/rollback plan).

---

## 6. Open questions (for maintainer/architect)

1. **Name:** top 3-5 from the unique-name bucket to verify for GitHub/PyPI/trademark?
2. **Base:** fork from `Ex3c4Def/basalt_ada` (decided); pin the exact commit/hash to fork from and record it.
3. **First dense model & artifact source:** local NInfer artifact via bridge, or basalt-pack/GGUF?
4. **First sparse config:** which Flash-Next quant/profile for apples-to-apples vs Strata IQ4_XS?
5. **Minimum API/features for v0:** streaming, tool calls, vision, grammar, multi-slot concurrency?
6. **Host RAM assumption:** 31 GB (Strata-like) or 64 GB?
7. **Quality gate thresholds:** what KLE/perplexity/NIAH thresholds define "acceptable" at IQ4_XS?

---

## 7. Artifacts created (this review)

- `<private>/projects/active/basalt-ada-engine/RESEARCH_PREP.md` (research lift)
- `<private>/projects/active/basalt-ada-engine/REVIEW_AND_STAGING.md` (this file)
- Local shallow clones (inspection only): `~/wt/basalt-ada-lift`, `~/wt/basalt-upstream-lift`
- Ninfer prep4qwen research dossiers consulted (13 files on `prep4qwen` branch): ktransformers, sglang, vllm, tensorrt-llm, flashinfer, qwen38-flash-next, deepseek-v4-sm89-xltzsoft, eagle3-medusa-specforge, lmdeploy-turbomind, ninfer-4090-udpsendtofailed, ninfer-windows-natpate, qwen-official, tokenspeed.

*No implementation or build actions were taken.*  
*End of review.*  
