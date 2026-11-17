> [!NOTE]
> Internal design note from 2026-11-16, kept as written apart from host, path and name scrubbing. It predates the
> engine. `SPECS.md` and `RESEARCH.md` describe what was built, and the gates in `gates/` record each step.

# Audit log — basalt-ada-engine research artifacts

**Date:** 2026-11-16  
**Session:** design  
**Scope:** scrutinize all reports/notes created this session, verify contested data against primary sources, correct errors, and align into a build-ready spec.

---

## Artifacts reviewed
- `project.md`, `RESEARCH_PREP.md`, `REVIEW_AND_STAGING.md`, `SM120_REPLACEMENT_AND_RESEARCH.md`
- `../research/sm89-sparse-dense-engine-quality.md` (deep-research final report)
- `../research/sm89-sparse-dense-engine-quality/*.md` (kernel, dense, sparse, kv, quantization, iq4xs, spec-decode, validation, memory/serving, qwen4 switch)

## Method
Read every artifact in full. For contested claims, fetched primary sources directly: Qwen3.8-27B `config.json`, Qwen3.8-Flash-Next `config.json`, UDPSendToFailed/ninfer-4090 README, PolyStrata `HOW_IT_WORKS.md`. Cross-checked local clones (`~/wt/basalt-ada-lift`, `~/ninfer-4090` branch `prep4qwen`) and local records.

---

## Errors found and corrected

### E1 — Sparse target model misidentified (material)
- **Wrong:** deep-research report §1 named the sparse reference as *Qwen3-Next / Qwen3-Coder-Next, 80B/3B active*.
- **Why wrong:** those are architecture analogues. The project's sparse target, and the model basalt's `qwen4exp` adapter implements, is **Qwen3.8-Flash-Next (`qwen4_exp`, 48 layers, 512 experts top-10+1 shared, QSA, PLE n-gram, gated residual)**.
- **Evidence:** HF `Qwen/Qwen3.8-Flash-Next` `config.json` (verified 2026-11-16) and `~/wt/basalt-ada-lift/src/models/qwen4exp`.
- **Fix:** rewrote report §1 with the verified Flash-Next config; retained Qwen3-Next as an explicit analogue. Dense config also replaced with verified Qwen3.8-27B `config.json` (was citing Qwen3.5-27B).

### E2 — KV cosine-vs-FP32 mislabeled (material)
- **Wrong:** `REVIEW_AND_STAGING.md` §4.2 and `SM120_REPLACEMENT_AND_RESEARCH.md` §3.3/§5 stated *rk4v4-e8 = 99.8% cosine*.
- **Correct (verified from UDPSendToFailed README):** int8 99.8%, rk8v4 99.4%, **rk4v4-e8 98.7%**, rk4v4 97.8%, **rk2v4-e8 96.2%**. The 99.8% belongs to INT8, not E8.
- **Impact:** the quality/capacity tradeoff is slightly larger than claimed; rk4v4-e8 is 1.1 pts below INT8, not equal. Both docs corrected. The deep-research report already had it right.

### E3 — PolyStrata marked as "not found" (gap now closed)
- Several notes listed PolyStrata as uncitable. Verified primary repo: `simplexai-labs/PolyStrata`, MIT, Strata-derived, tiered GPU/RAM/SSD experts, in-model MTP + prompt lookup. Added verified facts to `SM120_REPLACEMENT_AND_RESEARCH.md` §2.5 with the correct expert counts (512 for Qwen, 24,576 for GLM).

### E4 — Terminology: "KLE" (normalization)
- Project and user prose uses "KLE". Deep-research notes use **KLD / KL divergence**. These refer to the same thing: next-token distribution KL divergence vs a BF16 reference. Standardized on **KLD (KL divergence)** in `BUILD_SPEC.md`; "KLE" treated as the maintainer's shorthand for the KLD-based quality error.

---

## Gaps re-scoped (not errors, now explicit)
- No controlled, same-model/same-corpus sm_89 KLD/PPL leaderboard across engines exists in public sources. This is why `BUILD_SPEC.md` defines a project-local BF16 reference and acceptance bands rather than importing thresholds.
- No peer-reviewed "MoE quality preservation on Ada across tiers" study; local Strata/basalt operational evidence is the strongest available.
- Chinese-language primary papers on sm_89 kernel *quality* remain sparse; LMDeploy/FastLLM/PowerInfer/KTransformers provide implementation evidence, not BF16-parity studies.
- Several research notes cite local paths under `~/wt/ninfer-v3`, `~/wt/ninfer-qwen4`, `~/wt/ninfer-gemma4`; all verified to exist on disk.

## Attribution conflict flagged
- `basalt-ada-lift/CONTRIBUTING.md` line 49 requires "Ported or adapted work keeps its original authors as `Co-Authored-By` lines." Maintainer global rule forbids `Co-Authored-By` attribution trailers. **Resolution in `BUILD_SPEC.md`:** preserve original authors in LICENSE/THIRD_PARTY_LICENSES and source headers (MIT obligation), but emit **no** git `Co-Authored-By` trailers.

## Update 2026-11-16 (late): Qwen4 family forward-compat
- Maintainer noted Qwen3.8 is the target but Qwen4 is the future (Qwen4-Max out now; "27b and next will be soon").
- Verified: Qwen4-Max released 2026-11-16, **closed-weight, API only** — not a local engine target. 2.4T backbone / 71B active / 328B n-gram / 1M context; architecture = GDN+QSA hybrid + gated residual + n-gram embedding, i.e. exactly what Flash-Next (`qwen4_exp`) previewed. Qwen4-27B/Next have no weights, card or date as of 2026-11-16; open weights promised later, no date.
- Consequence added to `BUILD_SPEC.md` §1.4: build component/config-driven adapters (GDN, QSA, gated residual, PLE n-gram, MTP, ViT) so a future Qwen4 checkpoint is a config + weight-conversion change, not a new kernel port. Do not bake conflicting blog parameter counts into code; HF `config.json` is authoritative. YaRN 1M is the context lever, gated by depth-sensitive quality validation.

## Status
All four errors corrected in place; Qwen4 forward-compat added. `BUILD_SPEC.md` is the single aligned, build-ready source of truth.

## Refinement pass 2026-11-16 (Qwen4, sparse use case, dense/sparse switch)
- New sources checked directly: Qwen4-Max blog `qwen.ai/blog?id=qwen4-max` (page dated 2026/11/15, GA reported 2026-11-16) and SGLang PR 36968 (real L20 and RTX 4090 sm_89 QSA evidence). Research base notes `doc:prep4qwen/qwen38-flash-next`, `doc:prep4qwen/qwen-official`, `doc:prep4qwen/targets` used with their labels: T3, status draft.
- Qwen4-Max facts in §1.4 confirmed and tightened: 2.4T backbone / 71B active / 328B n-gram, GDN+QSA, four-branch GR, layer-2 n-gram embedding, API only, smaller open models promised with no size, date or license. No correction needed to the earlier forward-compat entry.
- New T4 lead labelled, not promoted: Qwen4-27B as dense 27B with hidden 5120, four-token MTP and QSA for 24 GB comes from keynote slides and kernel-code leaks, not a config or model card. It stays out of spec inputs.
- New finding from the fork's code, not a research claim. `tools/check_model_layering.sh` forbids one model from including another, and all GDN/QSA/GR/PLE/ViT kernels live in `src/models/qwen4exp/kernels/`. The earlier phrase "a dense adapter that reuses the GDN/full-attention/ViT machinery currently inside qwen4exp" was not implementable as a header include. `BUILD_SPEC.md` §1.6 now states the boundary: promote those components into a config driven shared layer first, then write thin adapters. Decision 8 records this. No prior corrected claim was overturned.
- Added `../research/sm89-sparse-dense-engine-quality/sm89_qwen4_sparse_vs_dense_switch.md` with the detail, the dense versus sparse decision table and the unverified list. `BUILD_SPEC.md` §1.4 to §1.6 and §6 updated; aligned sources now counts 9 research notes.
- sm_89 stayed the only target throughout. No sm_86, sm_90, sm_100, sm_120 or multi-arch claim was added.

## Research lane 2026-11-16 (IQ4_XS quality and performance), doc-only
- One research lane produced `../research/sm89-sparse-dense-engine-quality/sm89_iq4xs_performance_quality_lane.md`. No code, build, GPU job or conversion was done.
- Lane verdicts kept: direct GGUF ingest with bytes unchanged; per-encoding kernel table (Q5_0 down experts are the uncovered row); reject a 4-bit tensor-core weight path because IQ indices need codebook expansion; reject re-quantizing to a groupwise NInfer format for v1; keep the PLE table at 4 bits.
- Lane challenges promoted into `BUILD_SPEC.md`: §3.1 added for the mixed-encoding artifact; §3 row 4 and row 9 corrected; §4.1 sparse BF16 fallback; §4.2 split of engine-numerics gate versus artifact comparison; §5 P0 host RAM and artifact identity; decision 9; §7 items resolved.
- Correction to the lane itself. The note claimed 84.5 GiB / 90.7 GB on disk for the QFN artifact. The follow-up pass checked all four shards with `stat`: 93,672,359,738 bytes, 87.24 GiB, 93.67 GB. The note now carries a verification block. The per-encoding percentages are marked lane-reported T0; a separate reparse in the follow-up pass could not read one GGUF metadata field and was not pursued. The spec uses the verified size and the mixed-encoding structure, not the lane's absolute byte total.
- Unverified, recorded as open: host-DRAM bound decode hypothesis, KLD share attributable to IQ4_XS blocks alone, int8 activation KLD cost, PLE table quality per byte, Dynamic v3.0 to v3.1 diff, which revision the published 93.7 GB card row measured, GSQ/RCO at equal bytes.

## Scope narrowing 2026-11-16 (maintainer, doc-only)
- The engine replaces NInfer and Strata for main-model serving; llama-swap stays as the only front door. It is Qwen dense and Qwen sparse only, tuned for the Qwen4 family. No GLM, no 35B class, no DeepSeek, no other families. Qwen4-27B is the intended main model; Qwen3.8-27B is a temporary dense proxy because Qwen4-27B still has no weights (T4 lead only).
- Provenance fields checked directly on the live QFN shard: `general.quantized_by = Unsloth`, `general.repo_url = https://huggingface.co/unsloth`, `general.quantization_version = 2` (GGUF format field, not the Dynamic generation), `general.file_type = 30`, `quantize.imatrix.dataset = unsloth_calibration_Qwen3.8-Flash-Next.txt`, `quantize.imatrix.file = Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf`, chunks 45, entries 926. Unsloth's Dynamic 3.1 page lists both Qwen3.8-27B and Qwen3.8-Flash-Next.
- `BUILD_SPEC.md` updated: §0 scope and replacement target, §1.1 target versus proxy, §1.3 out-of-scope boundary, §1.7 artifact hardlock, P4 and P6 gates, decisions 10 and 11, and removal of the stale 31 GB phrase in §1.5. No earlier correction overturned.
- Local dense constraint: only the `MTP/` head dir for `unsloth/Qwen3.8-27B-GGUF` is present; main v3.1 shards must be fetched and hashed in P0. The old NInfer `.ninfer` dense artifact is not accepted by the new engine.

## Final review and handoff 2026-11-16 (doc-only)
- Full read of `BUILD_SPEC.md` found inconsistencies introduced by the layered refinements. Fixed: §0 pointed at §5 for the quality contract instead of §4; §1.2 still called qwen4exp the "basalt adapter"; decision 4 still named Qwen3.8-27B as the dense model rather than the proxy; decision 7 still said FP32 accumulation generally; P5 still said correctness vs BF16; §7's resolved-item note pointed at the wrong decision; §1.7 named only the two current models; P0 asked for only the sparse BF16 fallback.
- New finding from the review, not from the earlier lanes: BF16 reference on one 4090 is also impossible for the dense proxy. Qwen3.8-27B at BF16 is about 54 GB. §4.1 now requires a dense fallback alongside the sparse one, P0 asks for both, §7 item 3 asks for both, and P4's gate says "declared dense reference". Every remaining "vs BF16" phrase now means the declared reference.
- Added the private handoff as the reading guide and ordered work queue for the implementer. It lists locked decisions, the six maintainer decisions that block P1, the seven phases and gates, verified local facts, known traps, evidence locations and explicit exclusions. The spec header points to it. No prior corrected claim was overturned.
- Checks: section order and numbering verified, stale phrase grep clean apart from the intentional explanation of why 31 GB is wrong, `git diff --check` clean.

## 2026-11-16 review 2 (the implementer, Fable 5.2)

Refinement, not a rewrite. Added BUILD_SPEC §9 and matching lines in HANDOFF. Reasons:
- Maintainer clarified the base: Basalt code plus the best of Strata 0.2.10, in our own repo with no upstream at all. §6.1 allowed optional upstream cherry-picks; §9.1 removes them.
- Verified the `basalt_ada` checkout (`499fad2`, v0.3.0) is still sm_120-only in `CMakeLists.txt`. Earlier docs called this "not evidenced"; it is now checked.
- Strata 0.2.10 provenance disagrees across `COMMIT` (`af6224f3`), `RELEASE.txt` (`eff3155`, tag `db0500a`) and the eval README (tag `ffa80d73`). `af6224f3` matches the branch head in `~/strata-dev`.
- §0 said speed is a constraint. The maintainer asked for best KLE, decode and prefill, so §9.4 adds numeric floors from local measurements.
- §7.3 had no mechanism for the BF16 reference. Basalt's `docs/precision/quality.md` gives one (`llama-perplexity --kl-divergence-base` on Unsloth BF16 GGUFs); adopted in §9.5.
- KV byte arithmetic in §9.5 is derived from the two configs, not measured. The dKLD values are Basalt's on sm_120, quoted, to be re-measured.
- `/mnt/models` at 98% is a new P0 blocker (§9.7).
- CUDA 13.4.92 verified, so the "relax the floor" clause in §2 is moot.

## 2026-11-16 review 3 (the implementer, Fable 5.2)

- **Error in review 2, corrected.** §9.2 and §9.4 said `basalt_ada` had no Ada port. Only `main` was checked, on a shallow clone. Branch `fork_4090` (`24362905`) holds a working port with its own notes and 4090 test results. Marked inline and superseded by §10.1. Lesson: list remote branches before judging a fork.
- §3 row 4 ("NVFP4 compiled out on sm_89") contradicts the fork, where NVFP4 runs through MMQ. Noted in §10.1.
- Base fixed by maintainer: latest Basalt. Recorded as `fork_4090` merged with `v0.4.1` (`d3b75742`), two conflicts on a trial merge.
- Upstream 0.4.0 added a dense stack and a Qwen3.8-27B adapter; §1.3, §1.6, decision 4 and P4's "new dense adapter" are overtaken. §10.3 restates the remaining dense work.
- Maintainer: nothing for DeepSeek-V4.1-Flash. §10.4 deletes it at import.
- Maintainer: web UI is metrics and model stats only, own design, Basalt's level of detail. §10.6. Field inventory taken from element ids and keys in `serve/web/` at v0.4.1, not from a running page.
- BF16 bases are published (`jesdga/basalt-kld-bases`); §10.5.
- Phase order changed to sparse before dense, §10.7.

## 2026-11-16 maintainer decisions (§11)

- Dense weights: maintainer asked for the most precise option and assumed NInfer converts the GGUF. It does not: it requantizes BF16 safetensors with plain absmax groupwise formats (`docs/weight-conversion.md`). Chosen: lossless repack of the Unsloth GGUF, proven per tensor.
- Drafter: DFlash2 against MTP A/B on both paths. Drafters admitted by manifest; the Unsloth rule covers target weights.
- Multi-GPU and the in-engine router removed.
- Name shortlist checked against PyPI, crates.io, npm, GitHub, Hugging Face and web search. No formal trademark search was run; Tor web search returned little, so "no conflict found" is weak evidence.
