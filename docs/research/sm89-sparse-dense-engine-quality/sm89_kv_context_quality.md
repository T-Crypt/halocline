# sm_89 KV cache and context quality (Ada, BF16 reference)

Scope: KV cache formats/quantization and context behavior on NVIDIA Ada `sm_89` (RTX 4090 class), with BF16 treated as the quality reference. Focus is quality/capacity/context behavior, not raw throughput except where it explains a quality tradeoff. Facts below are cited to primary local measurements/docs or external sources; unverified areas are listed in **Gaps**.

## 1. What KV cache formats are available on sm_89 (Ada, 24 GB class)?

### Takeaway
On sm_89, BF16/FP16 KV is the baseline reference but is capacity-limited; practical low-bit options in use are INT8/FP8 and rotated/E8-compressed key/value modes in specialized engines (notably ninfer-4090), while some Blackwell-oriented formats (e.g. NVFP4 KV paths in certain stacks) are not available on RTX 4090 in the ninfer-v3 fork. General engines expose different subsets (llama.cpp `f16/bf16/q8_0/q4_0/...` KV types; vLLM FP8 KV cache with backend/arch constraints).

### Cited Findings
- NInfer-4090 (sm_89 fork) documents KV storage options `bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8`, but states `nvfp4` and `k8v4` are upstream `sm_120a` modes not available on the RTX 4090 in this fork; `rk4v4-e8` is recommended for long context on 4090. — `~/wt/ninfer-v3/docs/cli.md`
- The same fork ships E8/rotated KV modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`) and a 262K-to-1M visible-keys envelope lift as ports from `UDPSendToFailed/ninfer-4090`, with mode-specific runtime-per-token footprints measured on the merged branch (e.g. ~26.4 KB/token `rk4v4-e8`, ~19.0 KB/token `rk2v4-e8`, ~35.9 KB/token `int8` slope). — `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- NInfer-4090 README reports sm_89 deployed profiles trading KV precision/capacity: text-only MTP3 `rk4v4-e8` at 262,144 tokens (5.08 GiB KV runtime, 1.37 GiB slack), `rk2v4-e8` at 262,144 (4.01 GiB, 2.43 GiB slack), and `int8` at 172,032 (6.31 GiB, 136 MiB slack); INT8 172032 starts while 196608 is rejected at startup by a byte-exact deficit. — `~/wt/ninfer-v3/README.md`
- `UDPSendToFailed/ninfer-4090` README (feat/rtx-4090-sm89-native) advertises INT8, Hadamard-rotated (`rk4v4`, `rk8v4`), and E8 lattice/cylinder modes (`rk4v4-e8`, `rk2v4-e8`) on a 24 GB RTX 4090, with mode footprints 260/196/132/136/100 B/token respectively and binary-searched physical ceilings (e.g. text-only MTP0: 567K `rk2v4-e8`, 433K `rk4v4-e8`, 294K `rk8v4`, 223K `int8`). — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md)
- llama.cpp exposes quantized/float KV cache types via `-ctk`/`-ctv` (e.g. `f16`, `bf16`, `q8_0`, `q4_0`, etc. in upstream/docs usage); NInfer-v3 depth comparison measured llama.cpp on Qwen3.8-27B `UD-Q4_K_XL` with `q8_0` KV cache and flash attention. — `~/wt/ninfer-v3/docs/llamacpp-comparison.md`
- vLLM documents FP8 KV cache support (`--kv-cache-dtype fp8` and related calibration/scaling paths) as a feature, with architecture/backend-specific behavior and caveats documented in its FP8 quantization/KV-cache feature docs. — [vLLM FP8 quantization docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- FlashInfer provides attention/KV-cache kernels and APIs spanning paged/cascade layouts and multiple backends/precisions (including FP8-related paths in supported configurations), but feature availability is architecture- and kernel-path-specific. — [FlashInfer docs](https://docs.flashinfer.ai/)

### Inferences
- On a 24 GB sm_89 card, format choice is primarily a quality-vs-resident-context trade: BF16/INT8 maximize precision headroom but cap context, while 4-bit/2-bit rotated-E8 modes recover the model's native 256K+ envelope at some measured quality/perf cost.
- "Available on sm_89" is engine- and kernel-path-dependent, not a single universal matrix: a dtype name can exist in CLI/config while being gated out for Ada in a given fork/backend.

### Gaps
- No single authoritative cross-engine matrix was found for all sm_89 KV dtypes (BF16/FP16/FP8/INT8/INT4/NVFP4 variants) with identical model/workload/quality metrics.
- SGLang-specific sm_89 KV format/quality support details were not collected from primary docs in this pass (see Q6 gaps).

## 2. What are the quality tradeoffs per KV format on sm_89 (BF16 reference)?

### Takeaway
Measured ninfer-v3/ninfer-4090 data on sm_89 show a clear gradient: INT8 is closest to the high-precision baseline in measured retrieval/drafting behavior but costs capacity; `rk4v4-e8` preserves retrieval/MTP-acceptance quality very well at much larger context; `rk2v4-e8` remains retrieval-usable in reported probes but sits lower on cosine/quality margin and costs more decode. Lower-bit cache quantization can be much more decision-sensitive than weight-only changes, consistent with recent quantization-decision literature.

### Cited Findings
- NInfer-v3 README (RTX 4090, greedy, CUDA graphs, official Qwen3.8-27B artifact): moving default from INT8 KV to `rk4v4-e8` serves full native 262,144 context with "exact single-needle, 5-needle, and code-detail retrieval through 260K tokens", identical MTP acceptance at 111K depth (78.8% vs 78.4%), 5.7% decode tax on a shallow greedy code probe, and prefill within 1-2% at matched depth vs INT8. — `~/wt/ninfer-v3/README.md`
- `docs/udp-fork-comparison.md` merged-branch measurements (same card/probes): `int8` 172032 context with NIAH pass at 59k/118k; `rk4v4-e8` 262144 context with NIAH 59k-260k pass, 5/5 needles @118k, 3/3 code details @168k, decode 126.6 tok/s and MTP acc @111k 78.8% vs INT8 134.2/78.4%; `rk2v4-e8` 262144 context with NIAH 260k pass, 5/5 @118k, 3/3 @168k, decode 120.5 tok/s (MTP acc row not reported for rk2v4-e8 in that table). — `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- `docs/v3-port.md` records quick perplexity on the v3 parity setup (int8 KV, 4096/2048): `rk4v4-e8` 4.3624 and `int8` 4.3430, alongside near-parity full-corpus perplexity context (v3 4.651043 vs v2 A16 4.651185, -0.003%) for the port validation run. — `~/wt/ninfer-v3/docs/v3-port.md`
- `UDPSendToFailed/ninfer-4090` README reports cosine similarity vs FP32 by KV mode as 99.8% (`int8`), 99.4% (`rk8v4`), 98.7% (`rk4v4-e8`), 97.8% (`rk4v4`), 96.2% (`rk2v4-e8`), and 100% recall on 359,169-token NIAH with `rk2v4-e8`; it also recommends setting max context 20-30K below physical ceilings for sustained operation. — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md)
- arXiv 2608.06564 ("Which Decisions Low-Bit Quantization Breaks, and How to Predict Them") studies decision changes across 16 models/families at 4/3/2 bits and finds decision margins shrink approximately linearly under quantization (`m' ≈ c m + b`, slope usually < 1 and smaller at lower precision), with different decision types affected differently and no single 3-bit failure point across schemes/tasks. — [arXiv:2608.06564](https://arxiv.org/abs/2608.06564)
- NInfer-v3 README also shows `rk2v4-e8` passes text retrieval gates (single-needle 260K, 5-needle 118K, exact code details 168K) at ~10% decode tax on the shallow code probe, while buying slack/capacity tradeoffs in vision+context profiles rather than more text context than the 262,144 native limit. — `~/wt/ninfer-v3/README.md`

### Inferences
- For retrieval-heavy long-context workloads on sm_89, `rk4v4-e8` is the strongest measured quality/capacity point in ninfer-v3 evidence (near-INT8 retrieval/MTP behavior with 262K context), while `rk2v4-e8` is a higher-compression option whose risk is better characterized by task probes/cosine than by context length alone.
- Cosine similarity is a useful compression-quality indicator but not sufficient alone; ninfer evidence relies on depth probes (single/multi-needle, code detail, MTP acceptance) and the arXiv result indicates decision margins can flip even when aggregate behavior appears strong.

### Gaps
- No BF16-vs-format perplexity/NIAH sweep across all ninfer KV modes on identical corpora was found in local docs (only selective quick/full parity points and retrieval probes).
- sm_89-specific quality comparisons for FP8 KV in vLLM/FlashInfer backends (perplexity/NIAH vs BF16) were not directly measured in gathered sources.
- 2608.06564 is a general low-bit quantization decision-margin study, not an sm_89 KV-cache-only benchmark; mapping its slopes/flip rates to these exact KV modes requires engine-specific measurement.

## 3. How does E8 lattice KV quantization work, and what quality has it achieved?

### Takeaway
E8-based KV compression in ninfer-4090 encodes key (and in practice paired value) information using structured 8-dimensional lattice/root codebooks (Hadamard-rotated subspaces / Conway-Sloane or cylinder variants) to reduce key bits while preserving dot-product geometry better than naive low-bit scalar quantization; measured sm_89 results show high cosine and strong retrieval at long context, especially for `rk4v4-e8`.

### Cited Findings
- NInfer-v3 ports `rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8` KV modes and E8 codecs from `UDPSendToFailed/ninfer-4090`, verifying the E8 codec bit-exactly against the upstream microbenchmark (96.155% / 98.678% cosine in that validation note) while integrating into ninfer-v3's retuned sm_89 prefill schedule. — `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- The ninfer-v3 E8 port test/coverage notes document that `rk2v4-e8` uses the 240-root `E8Root` path while `rk4v4-e8` uses the `E8Lattice` path (`kv_e8_lattice`), and that production cache writes go through the warp encoder path; codec parity/tie behavior was characterized without rewriting stored `rk2v4-e8` codes. — `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- Local synthetic oracle `tools/test_kv/kv_oracle.py` demonstrates the mathematical mechanism used for evaluation: 256-d keys are split into 8-d subspaces, Hadamard-rotated, normalized/scaled, projected via Conway-Sloane E8 nearest-lattice search (D8 union D8+0.5), refined with a residual E8 stage, inverse-rotated/reconstructed, then evaluated by cosine similarity and multi-needle retrieval (asserts mean cosine >= 0.990 and 100% needle retrieval in that synthetic harness). — `~/wt/ninfer-v3/tools/test_kv/kv_oracle.py`
- `UDPSendToFailed/ninfer-4090` README describes E8 mode footprints and reports cosine vs FP32: `rk4v4-e8` (4-bit E8 lattice keys + 4-bit values, 136 B/token) 98.7%, `rk2v4-e8` (2-bit E8 cylinder keys + 4-bit values, 100 B/token) 96.2%, with 100% recall on 359,169-token NIAH using `rk2v4-e8` and long-context envelopes up to 567K (MTP0, text-only). — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md)
- NInfer-gemma4 dossier on the same sibling fork summarizes the implementation as Conway-Sloane E8 nearest lattice projection for 4-bit keys and 240-root S^7 factorization for 2-bit keys (H8 Hadamard rotation, multiplier-free dot decode, packed representation), fused into quantize/decode/attention/append paths rather than a drop-in transform. — `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`
- Independent/related E8 KV compression work exists: "KV Cache Compression using the E8 Lattice" (arXiv:2610.15384) proposes E8 lattice VQ for low-bit KV compression; `splickz/qwen3.8-27b-rk4v4-e8-sm120` is an HF artifact/model card referencing the E8 4-bit KV path for Qwen3.8/NInfer context; `jmarquex/nexusquant` and `FluffyAIcode/LLM-KV--Cache-compress` are related public repos/projects exploring E8/lattice-style KV compression ideas. — [arXiv:2610.15384](https://arxiv.org/abs/2610.15384), [splickz/qwen3.8-27b-rk4v4-e8-sm120](https://huggingface.co/splickz/qwen3.8-27b-rk4v4-e8-sm120), [jmarquex/nexusquant](https://huggingface.co/jmarquex/nexusquant), [FluffyAIcode/LLM-KV--Cache-compress](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)

### Inferences
- E8 KV quality gains over naive low-bit scalar quantization come from exploiting 8-D lattice structure and rotation/normalization to reduce quantization error in attention-relevant geometry, but realized quality depends on the exact codec variant (`rk4v4-e8` vs `rk2v4-e8`), integration point, and workload.
- The ninfer-v3 local oracle proves mechanism/fidelity properties in a synthetic setting; engine-level retrieval/MTP/perplexity probes are the relevant evidence for production quality claims.

### Gaps
- A public, architecture-neutral ablation isolating E8 lattice choice from all ninfer kernel/runtime effects on sm_89 was not found.
- No direct comparison was found between these ninfer E8 modes and other E8 KV implementations (e.g. NexusQuant/KakeyaLattice repos, arXiv 2610.15384 reference implementation) on the same Ada hardware/model/tasks.

## 4. How does YaRN extend context, and what is its quality impact?

### Takeaway
YaRN extends RoPE-based context windows by rescaling/interpolating rotary frequencies (with attention-temperature-related scaling in practical implementations) so models can extrapolate beyond trained length with much lower fine-tuning cost than prior methods; quality impact is task- and implementation-dependent, but it is a key enabler for very long contexts (including 1M envelopes in ninfer-4090 forks) rather than a guarantee of BF16-equivalent long-range reasoning.

### Cited Findings
- YaRN (arXiv:2309.00071) presents an efficient RoPE context-extension method for transformer LMs, reporting that it requires 10x fewer tokens and 2.5x fewer training steps than previous methods, enables LLaMA models to utilize/extrapolate to much longer context lengths than original pretraining, and surpasses prior state-of-the-art context-window extension results reported in that work. — [YaRN: Efficient Context Window Extension of Large Language Models (arXiv:2309.00071)](https://arxiv.org/abs/2309.00071)
- `UDPSendToFailed/ninfer-4090` README states direct L1-cached GQA decode block table lookups support a native context envelope up to 1,048,576 (1M) tokens and its dossier notes YaRN 1M frequency scaling plus attention temperature scaling were added in that fork's evolution (v1.1.0 lineage described in the dossier). — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md), `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`
- NInfer-gemma4 dossier summarizes ninfer-4090's YaRN 1M machinery as 3-band inverse-frequency interpolation plus attention temperature scaling anchored at 1M with runtime-updated frequency tables/attention scaling, and flags 2-bit KV long-context quality as not fully established beyond cosine/NIAH evidence. — `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`

### Inferences
- In the sm_89 ninfer context, YaRN addresses positional extrapolation/addressability while KV quantization addresses memory capacity; long-context quality is jointly constrained by both (and by the model's own training/behavior), so a 1M envelope does not imply uniform quality at 1M.
- For quality-focused deployments, YaRN should be validated with depth-sensitive probes (retrieval, code/detail, reasoning) at target lengths rather than inferred from theoretical extension claims alone.

### Gaps
- No ninfer-v3/ninfer-4090 sm_89 benchmark in gathered sources isolates YaRN's quality effect from KV dtype changes on the same model/prompts/lengths.
- Public task-level quality curves (beyond retrieval/NIAH claims) for YaRN-extended 256K-1M operation on this exact Ada stack were not found.

## 5. What NIAH (needle-in-a-haystack) results exist by context length and KV format?

### Takeaway
On sm_89 ninfer stacks, reported/measured retrieval remains strong well beyond typical INT8 capacity limits: `rk4v4-e8` preserves exact single-needle/5-needle/code-detail retrieval through 260K in ninfer-v3 measurements, while `rk2v4-e8` also passes reported retrieval gates in those probes and the sibling fork reports 359K NIAH success at extreme compression (with caveats about broader quality evidence).

### Cited Findings
- NInfer-v3 README: 64K and 128K NIAH exact answers are reported in the baseline results table (INT8-era rows shown), and the E8 default section states exact single-needle, 5-needle, and code-detail retrieval through 260K tokens for `rk4v4-e8`. — `~/wt/ninfer-v3/README.md`
- `docs/udp-fork-comparison.md` (merged branch): `int8` passes NIAH at 59k and 118k; `rk4v4-e8` passes NIAH at 59k-260k and achieves 5/5 needles @118k and 3/3 code details @168k; `rk2v4-e8` passes NIAH at 260k and also 5/5 needles @118k and 3/3 code details @168k. — `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- `UDPSendToFailed/ninfer-4090` README claims 100% (5/5 needles) recall on a 359,169-token needle-in-a-haystack with `rk2v4-e8` and reports a 360K NIAH line in its benchmark matrix/context discussion. — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md)
- NInfer-gemma4 dossier records the same sibling-fork claim (359,169-token NIAH, 100% recall on `rk2v4-e8`) while cautioning that cosine/NIAH are necessary but not sufficient for broader 256K agentic quality. — `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`
- NInfer-v3 llama.cpp comparison refresh reports E8 (`rk4v4-e8`) prefill/decode behavior at 176K/235K depths (NInfer-only territory vs compared llama.cpp config) but does not restate a full NIAH matrix in that section; retrieval claims are primarily in README/udp-fork-comparison. — `~/wt/ninfer-v3/docs/llamacpp-comparison.md`

### Inferences
- Retrieval robustness under compression is non-uniform by probe type: single-needle success can overstate quality, so ninfer-v3's inclusion of multi-needle and exact code-detail probes is materially stronger evidence for KV quality at depth.
- `rk2v4-e8` can retain targeted retrieval at very long contexts, but quality-sensitive non-retrieval tasks may degrade earlier than NIAH indicates.

### Gaps
- No standardized, cross-engine NIAH protocol on sm_89 (same haystack generator, depths, needle types, temperatures, prompts, and scoring rules) comparing ninfer, llama.cpp, vLLM, and SGLang was found.
- Public depth curves for many intermediate lengths/formats (especially BF16/FP8 in other engines on sm_89) are sparse or absent in gathered sources.

## 6. How do leading engines handle KV quality on sm_89 (NInfer, UDPSendToFailed ninfer-4090, llama.cpp, vLLM, SGLang/FlashInfer)?

### Takeaway
NInfer-family engines on sm_89 prioritize custom low-bit/structured KV (rotated/E8) with explicit retrieval/MTP validation to reach 256K+; llama.cpp emphasizes configurable KV dtypes with practical quality/speed tradeoffs and flash-attention constraints for some quantized paths; vLLM and FlashInfer expose FP8/optimized KV paths but sm_89 quality behavior is backend- and kernel-dependent and less directly documented as NIAH/perplexity matrices in the gathered sources. SGLang-specific sm_89 KV quality evidence is a gap.

### Cited Findings
- **NInfer-v3 / ninfer-4090 (this fork):** sm_89-targeted runtime with `rk4v4-e8` default for full native 262K context, measured near-INT8 retrieval/MTP quality, explicit rejection of oversized contexts at startup, and documented tradeoffs across `int8`, `rk4v4-e8`, `rk2v4-e8` (including vision scratchpad interactions). — `~/wt/ninfer-v3/README.md`, `~/wt/ninfer-v3/docs/cli.md`, `~/wt/ninfer-v3/docs/udp-fork-comparison.md`
- **UDPSendToFailed/ninfer-4090 (sibling fork):** pushes Ada-specific long-context capacity with E8/Hadamard KV modes, 1M envelope support, and reported high retrieval/cosine at extreme compression; dossier notes Windows/WDDM-centric headline ceilings/features and an agentic prefix-reuse issue (#15) relevant to context reuse behavior in tool loops. — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md), `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`
- **llama.cpp (comparison baseline in local measurements):** measured on same RTX 4090 with `UD-Q4_K_XL`, `q8_0` KV cache, flash attention, and batch settings matching deployment; reached ~144K practical ceiling in compared setup and showed prefill/decode/context tradeoffs vs ninfer E8/INT8 profiles. (General llama.cpp KV-type selection via `-ctk/-ctv` is supported; quality/speed depends on chosen K/V types and build/runtime features.) — `~/wt/ninfer-v3/docs/llamacpp-comparison.md`
- **vLLM:** provides FP8 KV cache support and quantization integration through its documented FP8 paths (`--kv-cache-dtype fp8` and calibration/scaling options), but supported kernels/features vary by GPU architecture and backend; sm_89-specific quality outcomes (NIAH/perplexity vs BF16) must be validated per chosen attention backend/config. — [vLLM FP8 quantization docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- **FlashInfer:** offers high-performance attention kernels and KV-cache-related APIs across multiple backends/layouts (including FP8-related functionality in supported paths), with architecture-specific kernel availability/performance characteristics; it is a kernel/backend layer whose end-to-end KV quality on sm_89 depends on engine integration choices. — [FlashInfer docs](https://docs.flashinfer.ai/)
- **Cross-engine context:** ninfer-v3 docs position E8 KV as the mechanism enabling full 262K native context on 24 GB sm_89 with bounded measured quality costs relative to INT8, in contrast to compared llama.cpp configuration limits in those tests. — `~/wt/ninfer-v3/README.md`, `~/wt/ninfer-v3/docs/llamacpp-comparison.md`

### Inferences
- Leading approaches split into (a) engine-specific structured KV compression tuned for capacity/quality (ninfer forks) and (b) standardized low-bit KV types integrated into general inference stacks (llama.cpp, vLLM/FlashInfer paths), with the former currently better documented for sm_89 long-context retrieval at extreme compression in this evidence set.
- Engine-level KV quality on sm_89 is not determined by dtype label alone; backend/kernel selection, prefix/cache reuse semantics, and context-management features materially affect observed behavior.

### Gaps
- No primary, citable sm_89 KV-quality benchmark set for **SGLang** (format support, BF16/FP8/INT8 comparisons, NIAH/perplexity) was gathered in this pass.
- vLLM/FlashInfer sm_89 quality claims here are constrained to documented capability/feature presence, not matched-quality measurements against BF16 on the same model/tasks.

## 7. What papers, repos, and other sources cover KV quality on Ada (including Chinese/community sources where available)?

### Takeaway
Relevant sources span (1) sm_89-focused engine implementations with measured long-context retrieval (ninfer forks), (2) low-bit quantization theory/decision studies (arXiv), (3) RoPE extension method (YaRN), and (4) E8/lattice KV compression papers/repos/cards; direct Ada-only, standardized KV-quality leaderboards remain limited.

### Cited Findings
- **Primary sm_89 engine/repos (local + upstream):** `T-Crypt/ninfer-v3`/ninfer-4090 docs and records document sm_89 Ada tuning, E8 KV modes, 262K deployment, and retrieval/MTP/perplexity measurements. — `~/wt/ninfer-v3/README.md`, `~/wt/ninfer-v3/docs/udp-fork-comparison.md`, `~/wt/ninfer-v3/docs/cli.md`, `~/wt/ninfer-v3/docs/v3-port.md`, `~/<private>/reference/records/ninfer-v3-port.md`
- **Sibling/community sm_89 fork:** `UDPSendToFailed/ninfer-4090` provides aggressive KV compression modes (E8/Hadamard), 1M envelope claims, cosine/NIAH results, and Windows/WDDM-oriented capacity features; dossier captures caveats (self-reported metrics, platform specifics, prefix-reuse issue). — [UDPSendToFailed/ninfer-4090 README](https://raw.githubusercontent.com/UDPSendToFailed/ninfer-4090/feat/rtx-4090-sm89-native/README.md), `~/wt/ninfer-gemma4/docs/research/ninfer-4090-udpsendtofailed.md`
- **Papers:** arXiv 2608.06564 analyzes how low-bit quantization breaks decision margins and predicts flip rates (relevant to KV/cache quantization sensitivity), while arXiv 2309.00071 (YaRN) provides the core RoPE extension method used to reach longer contexts efficiently. — [arXiv:2608.06564](https://arxiv.org/abs/2608.06564), [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- **E8 KV compression papers/repos/cards:** arXiv 2610.15384 proposes E8 lattice VQ for low-bit KV; public artifacts/projects include `splickz/qwen3.8-27b-rk4v4-e8-sm120` (HF model card/artifact referencing rk4v4-e8 path), `jmarquex/nexusquant`, and `FluffyAIcode/LLM-KV--Cache-compress`. — [arXiv:2610.15384](https://arxiv.org/abs/2610.15384), [splickz/qwen3.8-27b-rk4v4-e8-sm120](https://huggingface.co/splickz/qwen3.8-27b-rk4v4-e8-sm120), [jmarquex/nexusquant](https://huggingface.co/jmarquex/nexusquant), [FluffyAIcode/LLM-KV--Cache-compress](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)
- **Engine/framework docs (KV features/backends):** vLLM FP8 KV cache documentation and FlashInfer documentation describe supported KV/attention features and architecture-dependent backends relevant to Ada integration choices. — [vLLM FP8 quantization docs](https://docs.vllm.ai/en/latest/features/quantization/fp8.html), [FlashInfer docs](https://docs.flashinfer.ai/)

### Inferences
- The strongest Ada/sm_89-specific long-context KV-quality evidence currently found is implementation-centric (ninfer forks) rather than standardized benchmark literature, so cross-source comparisons require careful normalization of model, prompt, depth, temperature, and probe type.
- E8 KV compression is an active cross-community direction (papers, model cards, repos) that has already produced deployable sm_89 artifacts, but portability and quality guarantees vary by implementation.

### Gaps
- Chinese-language primary sources specifically reporting sm_89 KV quality/NIAH/perplexity measurements were not identified and cited in this pass.
- No comprehensive, peer-reviewed Ada-only (sm_89) KV-cache quality leaderboard comparing ninfer/llama.cpp/vLLM/SGLang under a shared protocol was found.

## Consolidated gaps (cross-cutting)
- **BF16-reference baselines on sm_89:** limited direct BF16-vs-all-KV-format sweeps on the same model/tasks/depths in gathered sources.
- **Perplexity coverage:** sparse beyond selective ninfer-v3 parity/quick numbers; no broad PPL matrix by format/length/engine on sm_89.
- **Standardized long-context probes:** NIAH/multi-needle/code-detail protocols differ across sources; SGLang and vLLM/FlashInfer sm_89 quality measurements are under-documented here.
- **YaRN isolation:** YaRN quality impact is entangled with KV compression/capacity in available sm_89 reports; isolated ablations are missing.
- **Chinese/community evidence:** targeted Chinese-source scan for Ada KV quality results was not completed in this pass.