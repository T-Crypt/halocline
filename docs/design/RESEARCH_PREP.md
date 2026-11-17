> [!NOTE]
> Internal design note from 2026-11-16, kept as written apart from host, path and name scrubbing. It predates the
> engine. `SPECS.md` and `RESEARCH.md` describe what was built, and the gates in `gates/` record each step.

# Basalt-Ada Engine — Research Lift & the implementer Prep

**Prepared:** 2026-11-16 (design session, no builds performed)  
**Project doc:** `project.md`  
**Purpose:** consolidate verified research, local context, and an execution-ready plan so an implementer can start implementation work with minimal rediscovery.

> **Constraint honored:** no binaries built, no services changed, no live configs edited, no GPU jobs run. Work performed: read/search/inspect, vault inventory over SSH, shallow clones for diff analysis.

---

## 1. Executive summary

- **Starting point is real but Ada support is unproven.** `Ex3c4Def/basalt_ada` (shallow clone inspected at `~/wt/basalt-ada-lift`, HEAD `1f0226cf`, 2026-11-06) is functionally identical in build target to upstream `jesdga95/basalt` (`~/wt/basalt-upstream-lift`, HEAD `ab345256`, 2026-11-14): both hard-require **Blackwell `sm_120` and CUDA 13+** (`CMakeLists.txt`), docs state NVIDIA Blackwell on Linux/Windows, README markets Blackwell. No `sm_89`/`4090`/Ada architecture enablement found in source/cmake (grep hits are metadata/tokenizer/test strings only).
- **`basalt_ada` is behind upstream.** `diff -rq` shows **601 differing paths** total; excluding `.git` pack internals, meaningful differences are mostly upstream-ahead additions/changes (e.g. `cmake/models.cmake`, `Dockerfile`, changelogs `0.3.2–0.4.1`, `docs/models/qwen38-27b.md`, `docs/configuration/documents.md`, release/community workflows, and many `src/`/`include/`/`tools/` differences). Treat `basalt_ada` as a fork point/name, not an Ada-ready delta.
- **Local engines provide the actual Ada/offload playbooks.**
  - **NInfer-4090** (`~/ninfer-4090 -> /data/ninfer-4090`, Apache-2.0): mature `sm_89` dense+vision/MTP serving engine for Qwen-family (and Gemma4 line on ext branch per records). Proven 4090 patterns: `rk4v4-e8` 4-bit KV for 262K on 24 GB, INT8 prefill path, paged KV/prefix reuse, CUDA graphs, cooperative-launch/runtime occupancy handling, prefill-chunk limits, strict VRAM/RAM budgeting, OpenAI/Anthropic serving, llama-swap integration. Key records: `reference/records/ninfer-v3-port.md`, `reference/records/ninfer-gemma4.md`, `guide/strata-and-ninfer.md`.
  - **Strata** (`~/strata`, MIT, fork line `T-Crypt/strata` deploy builds in `~/wt/strata-*`): MoE expert-offload reference for Qwen3.8-Flash-Next on 4090. Real numbers: ~15–16 GB VRAM + ~20.9 GiB pinned RAM, 262K q4_0 viable, 512K tight, sensitive to host RAM pressure (never build heavy beside resident Strata). `~/Strata-data` holds model data (~105 GB noted in docs); `~/strata-live` absent now.
- **Vault is sparse for this topic today.** `datavault` a lab container (`<lab-ip>`, reachable via SSH) inventoried: `/srv/vault/{research,datasets,rag,mcp}` empty; `/srv/vault/inbox/web/2026-11-15/huggingface.co-unsloth-qwen3-8b-gguf-hugging-face.md` contains only incidental `dense and mixture-of-experts` text (no Ada/sm_89/ninfer/strata/basalt engine material). **No vault artifacts currently change kernel/KV/model priorities.**
- **Recommended initial targets (for planning, to be ratified):** (1) **Dense:** Qwen3.8-27B class path (leverages NInfer `.ninfer` v3 artifacts/tooling knowledge and proven 4090 perf). (2) **Sparse/MoE:** Qwen3.8-Flash-Next (Strata/basalt native target; matches existing lab offload experience and artifacts). Treat Gemma4 as follow-on (vision/BF16-KV/no-prefix-reuse constraints differ).
- **Biggest risks:** basalt codebase is Blackwell/SM120-oriented (kernels, pack/quant assumptions, CUDA 13 features); dense+sparse unified runtime abstraction; 24 GB VRAM + host-RAM ceiling; artifact/format strategy; license boundaries for direct reuse from NInfer/Strata code.

---

## 2. Sources consulted (verified)

| Source | Location / evidence | Notes |
|---|---|---|
| Project doc | `<private>/projects/active/basalt-ada-engine/project.md` | Baseline scope, risks, phased plan |
| Basalt Ada fork (remote sampled earlier) | https://github.com/Ex3c4Def/basalt_ada | README/CMake/docs sampled; clone used for ground truth |
| Upstream Basalt | https://github.com/jesdga95/basalt | Clone diff baseline |
| Local clone: basalt_ada | `~/wt/basalt-ada-lift` (HEAD `1f0226cf`, shallow) | Build/docs/src/license inspected |
| Local clone: basalt upstream | `~/wt/basalt-upstream-lift` (HEAD `ab345256`, shallow) | Diff baseline |
| NInfer-4090 | `~/ninfer-4090 -> /data/ninfer-4090` (README, docs tree, LICENSE) | sm_89 engine, serving/KV/perf patterns |
| NInfer records | `<private>/reference/records/ninfer-v3-port.md`, `<private>/reference/records/ninfer-gemma4.md` | Port decisions, Gemma4 bring-up, runtime learnings |
| Strata | `~/strata` (README, LICENSE, configs/logs/releases present), `~/Strata-data` exists | MoE offload/profile; `~/strata-live` absent |
| Homelab guides/inventory/state | `guide/strata-and-ninfer.md`, `inventory/services.yaml`, `state/changes.md` (via lab lookup) | Ops rules, service facts |
| Data vault inventory | a lab container `datavault` <lab-ip>, `/srv/vault` (SSH, read-only listing/grep) | 2026-11-16 inventory result below |
| Lab lookup | `python <private>/scripts/lab/lab.py datavault|strata` | Entity card + recent changes |

**Not inventoried in depth (follow-ups):** full `~/ninfer-4090/src`/`apps` kernel internals, full `~/strata` engine source/build trees under `~/wt/strata-*`, full basalt `src/` kernel implementations (surveyed structure/entry points), lab research MCP contents beyond doc statement (MCP not queried programmatically here except context provided), and any external papers/docs beyond what’s already encoded locally.

---

## 3. Basalt / basalt_ada: verified findings

### 3.1 Build & platform targeting (blocking for Ada)
- `CMakeLists.txt` (both repos, sampled): defaults `CMAKE_CUDA_ARCHITECTURES` to `120`; **hard-fails unless arch base is 120**; **requires CUDA >= 13.0**. Message: *"basalt builds for compute capability 12.0 only"*.
- `cmake/platform.cmake`: ISA dispatch for host CPU paths (AVX512/AVX2/AVX/VNNI variants) — portable concern, but GPU side is Blackwell-only.
- `docs/platforms.md`: states *"Linux x64 and Windows x64, NVIDIA Blackwell (sm_120). ... Windows limitations ... quality not measured on Windows..."*
- `docs/building.md`, `README.md`: Blackwell/sm_120 positioning, `.basalt` packs for Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash.

### 3.2 Repo delta (ada fork vs upstream)
- **Same license:** MIT, Copyright (c) 2026 Jesús Gulfo (both).
- **Fork appears stale relative to upstream head (by content):** upstream adds `cmake/models.cmake`, Docker artifacts, more changelogs/docs (incl. `qwen38-27b`), extra workflows (`release.yml`, `community.yml`, actions dir), and numerous source diffs (601 path diffs total including `.git` noise). Example meaningful diffs seen: `CMakeLists.txt`, `cmake/platform.cmake`, `CONTRIBUTING.md`, `AGENTS.md`, many `docs/*`, `include/basalt/artifact/dequant.hpp`, workflows.
- **Conclusion (maintainer decision 2026-11-16):** base on `Ex3c4Def/basalt_ada` (fork and port directly). Upstream `jesdga95/basalt` is newer in content but the breakaway identity starts from the basalt_ada fork; upstream improvements can be cherry-picked later if needed. In either case expect substantial architecture-port work; do not assume the ada fork contains hidden Ada code.

### 3.3 Architecture cues (from file layout, not deep kernel audit)
- **Model adapters:** `src/models/qwen4exp/*` (large: load/prefill/decode/serve/kernels, many CUDA `.cu`), `src/models/deepseek41/*`, `src/models/builtin.cpp`; registry in `src/model/`.
- **Subsystems visible:** artifact/ggml-derived dequant & MMQ pieces (`include/basalt/ggml/`, `src/artifact/*`), tokenizer/grammar, serve (`serve/` Python per docs/README), tiers/offload concepts implied by config/docs and naming (expert residency/prefill/peer/cpu paths in qwen4exp prefill files like `peer_experts.cpp`, `cpu_experts.cpp`, `expert_fp4.cpp`, `moe_fused*`), CUDA graphs/bench/probe tools (`basalt-bench`, `basalt-probe`, `basalt-vision`).
- **Quant/KV sensitivity likely high:** presence of `expert_fp4`, `mmq`, `kv_q4.cu`, fused MoE kernels, sm_120 build gate — Blackwell-specific assumptions (TMA/mbarrier shapes, FP4/NVFP4 paths, cooperative launch params) must be audited like NInfer v3 port did.
- **Format/tooling:** custom `.basalt` packs + converter (`tools/convert/*`), inspect/probe, profiles under `data/profiles`.

### 3.4 Quick ripgrep signals (for the implementer to continue)
Useful searches already run on clones:
- `rg -n "sm_120|sm_89|compute_120|compute_89|__CUDA_ARCH__|TMA|mbarrier|nvfp4|fp4|e2m1" basalt-ada-lift/src basalt-ada-lift/include` (not fully dumped here; run locally)
- `rg -n "expert|offload|tier|peer|cpu_expert|pinned|ram" basalt-ada-lift/src/models/qwen4exp`

---

## 4. NInfer-4090: reusable Ada lessons (do not assume code-copy is allowed)

**Path:** `~/ninfer-4090` -> `/data/ninfer-4090` (Apache-2.0). **Runtime/records are the primary value for this project.**

- **Proven 4090/sm_89 choices (from README/records):**
  - E8 4-bit KV (`rk4v4-e8`) enables ~262K context on 24 GB with measured small decode tax vs INT8; INT8 KV for max precision at lower context.
  - INT8 prefill path on sm_89 for perf; fp8/plain paths used where appropriate; **NVFP4-family compiled out on sm_89** in port work.
  - Runtime cooperative occupancy via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (static sm_120a constants overflow Ada).
  - Prefill chunking matters (e.g. 1024 typical), auto-long-anchors/prefix reuse tradeoffs, host-state memory costs (GDN state images sizable).
  - Single-GPU, startup-fixed capacity, careful VRAM accounting; `--max-concurrency` cost model documented.
- **Serving contract:** OpenAI/Anthropic compatible HTTP, metrics, queueing/pending timeout considerations for deep prefills, MTP speculative decoding, reasoning/tool-call/vision (model-dependent).
- **Port discipline (v3 record, highly relevant to basalt):** 3-way merge at tip granularity for arch carry, superset merged catalogs/dispatch tables, arch-gate PDL/TMA/mbarrier intrinsics, per-file arch qualification by measurement, avoid `-j` storms and VRAM contention during tests. Treat Blackwell-only features (NVFP4, TMA usage patterns, cluster fences, k8v4 e2m1 KV) as excluded/redesigned on Ada unless explicitly ported.
- **Implication for basalt:** expect to (a) parameterize arch in CMake and all kernel dispatch, (b) replace/disable sm_120-only intrinsics & quant paths, (c) revalidate attention/KV/MoE kernels and graph capture on sm_89, (d) build a 4090 memory budget model before chasing 512K claims.

**License note:** NInfer is Apache-2.0 (per `~/ninfer-4090/LICENSE` header). Direct code reuse into an MIT project is possible in principle but requires preserving notices/attribution and understanding file provenance (some code derives from other projects per NInfer README credits). **Do a file-level provenance audit before copying.**

---

## 5. Strata: sparse/offload reference

**Path:** `~/strata` (install/config/logs/releases 0.1.43–0.2.13.1 present), configs like `strata-ud-iq4_xs*.json`; **data:** `~/Strata-data` (exists). **Source/worktrees:** `~/wt/strata-*` present in `~/wt`. **License:** MIT (`~/strata/LICENSE`, Niko1221 and contributors).

- **Operational profile (the workstation, 4090):** MoE expert-offload for Qwen3.8-Flash-Next; ~15–16 GB VRAM + ~21+ GiB pinned RAM; 262K q4_0 typical good point, 512K tight; decode can be strong at depth but host RAM pressure is critical. **Rule:** stop/unload competing GPU users; avoid heavy parallel builds when resident (OOM risk noted historically).
- **Run shape:** hand-run (`strata`, `strata stop/status/bench`), loopback `:8080/v1`, never alongside llama-swap. llama-swap unload required before start.
- **Reuse value:** expert residency/tiering policy, RAM pinning behavior, calibration knobs, benchmark methodology on 4090, and migration baseline for parity/regression.
- **Caveat:** `~/strata-live` absent now (per `guide/strata-and-ninfer.md` layout notes evolved over time); rely on `~/wt` worktrees and `~/strata` run artifacts for concrete behavior.

---

## 6. Data vault inventory (2026-11-16)

**Target:** `datavault` a lab container, `<lab-ip>`, `/srv/vault` (lab lookup confirmed; SSH read-only commands run).

**Top-level:** `README.md`, `datasets/`, `inbox/`, `lost+found` (permission denied to grep), `mcp/`, `rag/`, `research/`.

**Findings:**
- `datasets/`, `mcp/`, `rag/`, `research/`: **empty** (no files found at depth <=3 aside from dirs).
- `inbox/web/2026-11-15/huggingface.co-unsloth-qwen3-8b-gguf-hugging-face.md`: only match for search terms (`ada|sm_89|4090|ninfer|strata|basalt|sparse|dense`) was generic text: *"suite of dense and mixture-of-experts (MoE) models"* — **not engine/Ada research material**.
- **No files matched** `sm_89`, `4090`, `ninfer`, `strata`, or `basalt` in accessible vault content.

**Actionable conclusion:** vault does not currently supply missing design inputs; proceed with local code/docs + targeted web/library research when implementation starts. Re-run inventory if maintainer adds material to vault.

**Suggested re-run command:**
```bash
ssh <lab-host> 'grep -RInE "ada|sm_89|4090|ninfer|strata|basalt|sparse|dense|kv.?cache|expert.?offload|moe" /srv/vault 2>&1 | head -120'
```

---

## 7. License / provenance memo (prep-level, not legal sign-off)

| Component | License (observed) | Implication | Action before code reuse |
|---|---|---|---|
| `Ex3c4Def/basalt_ada` / `jesdga95/basalt` | MIT (Jesús Gulfo) | Forkable base | Keep LICENSE/NOTICE chain; review `THIRD_PARTY_LICENSES` in repo (not fully audited here) for ggml/llama.cpp-derived pieces |
| `~/ninfer-4090` (`T-Crypt/ninfer-4090`, upstream lineage Neroued/Don-Chad noted in README) | Apache-2.0 | Concepts/measured behavior safe to reuse; source reuse requires attribution/NOTICE care | Identify exact files/origins before copying; prefer clean-room reimplementation for high-risk kernels unless provenance is clear |
| `~/strata` / `T-Crypt/strata` | MIT (Niko1221 and contributors) | Compatible for reuse with attribution | Verify any upstream/forked subcomponents and model-conversion scripts |
| Model weights/artifacts (HF, local `.ninfer`/GGUF/basalt packs) | Model/card-specific | Separate from engine license | Track provenance sidecars/hashes (NInfer practice) and respect model licenses in docs/tooling |
| ggml/llama.cpp fragments inside basalt/ninfer | Upstream MIT (commonly) | Must preserve notices | Audit `THIRD_PARTY_LICENSES` and embedded headers per file |

**Policy for this breakaway (recommended):** default to **concept reuse + measurements + clean implementations** for kernels/critical paths; allow direct code import only when (1) license compatible, (2) provenance and third-party notices are preserved, (3) sm_89 correctness/perf is validated. Record decisions in a future `LICENSE_AUDIT.md` or ticket comment.

---

## 8. Recommended technical decisions (to ratify before coding)

1. **Base repo choice (maintainer decision 2026-11-16):** fork and port **directly from `Ex3c4Def/basalt_ada`** (not upstream `jesdga95/basalt`). Inherit its git history/lineage; do not rebase onto upstream basalt. Upstream improvements can be cherry-picked later if needed, but the breakaway identity starts from the basalt_ada fork. Keep the new engine name early (project "breakaway").
2. **First bring-up pair (smallest end-to-end value):**
   - **Dense:** a model/artifact path already proven on NInfer 4090 (Qwen3.8-27B `.ninfer` v3 or a tightly-scoped GGUF bridge) to validate sm_89 correctness, KV, prefill/decode, serving.
   - **Sparse/MoE:** Qwen3.8-Flash-Next (the basalt/Strata target) to validate router/top-k + expert residency/offload on 4090.
   - *Alternative if artifact friction is too high:* start with one small dense model supported by chosen format to prove runtime, then add Flash-Next MoE.
3. **Canonical artifact/IR (decision needed):** evaluate **(A)** adopt/adapt `.basalt` pack pipeline (native to basalt), **(B)** consume `.ninfer`/GGUF via bridge for faster bootstrap, **(C)** internal IR with converters from both. Pick based on: converter cost, existing artifact availability, ability to encode dense+MoE topology/expert layout/KV/chat-template/vision, and validation (hash/inspect/dequant parity).
4. **Arch/toolchain strategy:** configurable `CMAKE_CUDA_ARCHITECTURES` defaulting to `89` (and optionally `86` later) on the workstation; isolate arch-specific kernels; detect/guard Blackwell-only PTX/features (`TMA`, cluster barriers, NVFP4/e2m1 paths) behind capability checks or separate TUs.
5. **Runtime split:** explicit **dense fast path** (no MoE overhead) and **sparse path** (router, expert cache/pin, VRAM/RAM/NVMe tiers, async promotion) sharing tokenizer/KV/scheduler/serving layers where safe.
6. **4090 budgets (design constraints, measured later):** 24 GB VRAM hard cap; assume ~20–23 GiB usable depending config; host RAM contention matters (Strata lesson). Require headroom guards and exclusive-GPU defaults.

---

## 9. Proposed file-level implementation plan (for the implementer)

> Organize as tickets/milestones. Each milestone must end with a reproducible check on the workstation (no concurrent GPU jobs) and recorded numbers.

### M0 — Discovery closure (docs/repo, no build required to finish planning)
- [ ] **M0.1 Fork decision record:** fork from `Ex3c4Def/basalt_ada` at pinned commit (maintainer decision); record rationale and upstream diff summary (`~/wt` clones available) for future cherry-picks.
- [ ] **M0.2 License audit memo:** enumerate reusable vs clean-room areas (kernels, loaders, serve, converters); capture `THIRD_PARTY_LICENSES` from basalt and any imported code provenance.
- [ ] **M0.3 Target matrix doc:** first dense + first sparse model, artifact format, required features (context targets, KV modes, quants, tool-calling/vision), and explicit non-goals for v0.
- [ ] **M0.4 Deep source survey notes:** map basalt modules (`src/model`, `src/models/*`, kernels, tiers/offload code paths, serve, tools/convert) and identify sm_120-specific sites (`rg` inventory committed as a note).

### M1 — sm_89 build skeleton
- [ ] **M1.1 CMake/toolchain:** make CUDA arch configurable (default 89), remove hard-fail on non-120, relax CUDA 13-only assumptions where incompatible with the workstation toolchain (verify installed CUDA), update `cmake/platform.cmake` messages/guards.
- [ ] **M1.2 Compile stubs/guards:** compile out or stub Blackwell-only kernels/paths behind `BASALT_ARCH_SM120`/capability macros; ensure a minimal CPU/host-side or tiny GPU smoke target builds.
- [ ] **M1.3 CI/dev ergonomics:** keep CPU tests buildable where possible; document local build commands for the workstation.
- **Checks:** configure+build selected targets on the workstation; `ctest -L cpu` (if applicable) passes; GPU smoke runs alone on 4090.

### M2 — Ada kernel & memory foundation
- [ ] **M2.1 Attention/KV correctness first:** port or reimplement prefill/decode attention + KV read/write for sm_89-supported dtypes/modes; define supported KV formats (start narrow, e.g. fp16/bf16/int8 as feasible).
- [ ] **M2.2 GEMM/GEMV/MMQ/quant paths:** validate or replace sm_120-tuned paths; measure vs CPU/reference; document unsupported quants explicitly.
- [ ] **M2.3 Memory management:** paged KV (if adopted), arena sizing, prefill chunk limits, graph capture constraints, OOM/headroom guards for 24 GB VRAM / limited RAM.
- **Checks:** kernel parity tests on 4090; greedy/logit spot checks on a tiny model/case; VRAM/RAM accounting logs.

### M3 — Dense end-to-end
- [ ] **M3.1 Loader/weights path:** implement/bridge for chosen dense artifact format.
- [ ] **M3.2 Dense runtime:** forward pass, sampling, context/KV growth, no MoE overhead.
- [ ] **M3.3 Minimal serve/CLI:** OpenAI-compatible generation path, streaming optional-but-planned, health/metrics basics.
- **Checks:** coherent text generation, stable greedy behavior, TTFT/decode/prefill baseline recorded.

### M4 — Sparse/MoE end-to-end
- [ ] **M4.1 Router + expert execution:** top-k routing, expert GEMM/MMQ paths on sm_89.
- [ ] **M4.2 Tiered expert residency/offload:** VRAM cache + pinned RAM (+ optional disk/peer later) inspired by basalt `tiers` and Strata policy; async prefetch/promotion.
- [ ] **M4.3 Dense+sparse unification:** shared scheduler/KV/serving abstractions with mode-specific hot paths.
- **Checks:** correctness on Flash-Next (or chosen MoE), throughput/latency vs Strata baseline where comparable, residency/eviction behavior under memory pressure.

### M5 — Productization & migration
- [ ] **M5.1 Serving/ops:** load/unload, metrics, config UX, llama-swap/Berth integration plan, safe defaults (exclusive GPU).
- [ ] **M5.2 Validation matrix:** per-model parity gates (greedy/ppl/logit as applicable), long-context/tool-call/grammar/vision coverage as supported, perf regression tracking.
- [ ] **M5.3 Docs & breakaway:** engine name, build/run docs for the workstation, supported configs, migration/rollback plan from Strata.

---

## 10. Suggested first commands for the implementer (read-only / analysis)

```bash
# Project + prep
sed -n '1,220p' <private>/projects/active/basalt-ada-engine/project.md
sed -n '1,260p' <private>/projects/active/basalt-ada-engine/RESEARCH_PREP.md

# Local references
sed -n '1,220p' <private>/guide/strata-and-ninfer.md
sed -n '1,220p' <private>/reference/records/ninfer-v3-port.md
sed -n '1,220p' <private>/reference/records/ninfer-gemma4.md

# Cloned baselines (created during this prep)
cd ~/wt/basalt-ada-lift && git log -1 --oneline
cd ~/wt/basalt-upstream-lift && git log -1 --oneline
diff -rq ~/wt/basalt-ada-lift ~/wt/basalt-upstream-lift | rg -v '^Only in .*/\\.git/' | head -80

# sm_120 / Blackwell-specific hotspots
rg -n "sm_120|compute_120|CMAKE_CUDA_ARCHITECTURES|CUDA 13|nvfp4|e2m1|mbarrier|tma|cluster" ~/wt/basalt-upstream-lift

# Tiers/offload/MoE hotspots
rg -n "tier|offload|expert|moe|pinned|peer|ram|nvme|resident" ~/wt/basalt-upstream-lift/src ~/wt/basalt-upstream-lift/docs

# Vault re-check
ssh <lab-host> 'find /srv/vault -type f | head; grep -RInE "ada|sm_89|4090|ninfer|strata|basalt|sparse|dense" /srv/vault || true'
```

**Before any GPU test/build on the workstation:** ensure no other GPU job is running (`nvidia-smi`, check llama-swap/Strata state). Do not run heavy builds alongside a resident Strata process.

---

## 11. Open questions for the maintainer / architect

1. **Base & naming:** new repo under `~/`/org now, or work in-place in `~/wt`? Preferred engine name/binary names?
2. **Base commit:** pin `Ex3c4Def/basalt_ada` commit for the fork (fork decided; upstream basalt improvements cherry-picked later if needed).
3. **First dense model & artifact source:** use existing local NInfer artifact (`~/ninfer-v3-artifacts/...` / `~/ninfer-ext-artifacts/...`) via bridge, or target a basalt-pack/GGUF path first?
4. **First sparse target config:** which Flash-Next quant/profile mirrors current Strata use (`strata-ud-iq4_xs` lineage) for apples-to-apples comparison?
5. **Minimum API/features for v0:** streaming, tool calls, vision, grammar, multi-slot concurrency?
6. **Acceptable host-RAM footprint:** optimize for 31 GB box like Strata today, or assume 64 GB available?

---

## 12. Prep artifacts created

- `<private>/projects/active/basalt-ada-engine/RESEARCH_PREP.md` (this file)
- Local shallow clones for inspection (not part of private lab repo): `~/wt/basalt-ada-lift`, `~/wt/basalt-upstream-lift`

*End of prep. No implementation/build actions were taken.*  
