# How Halocline was built: the recipe

This file lists the steps, the gates and the sources behind Halocline, in the order I used them, so you can build an
engine for your own card the same way. Halocline itself stays sm_89 and Qwen4-family only (see the README); the method
here carries to any card.

Every paper listed below was checked against arxiv.org on 2026-11-17, and the titles are quoted as arXiv prints them.
The Hugging Face revisions are the ones the measurements used.

## The steps

1. **Pick one machine and one model family.** Write the machine down first (`SPECS.md`). Every later number belongs to
   that machine.
2. **Measure what you run today on that machine.** I measured Strata 0.2.10 (sparse) and NInfer-4090 (dense) with one
   smoke script on a quiet host, three runs each. Prompt speed on this host dropped 30 to 40% under memory pressure, so
   a quiet host is part of the method (`evals/2026-11-16/halocline-p0-floors/`).
3. **Freeze a quality reference.** The BF16 GGUF of the same model through `llama-perplexity --kl-divergence-base`
   (wikitext-2, 100 chunks of 512 tokens) gives a file of the full-precision next-token distribution. Hash it and never
   regenerate it. Halocline scores against
   [`flashnext-bf16-base.kld`](https://huggingface.co/datasets/jesdga/basalt-kld-bases), sha256
   `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9`, base PPL 3.2749 (3.2733 on the scored rows).
4. **Start from code that already runs on your architecture.** I took the basalt_ada `fork_4090` Ada port and merged it
   once with Basalt v0.4.1. After that snapshot there is no upstream: later releases get read, not merged.
5. **Remove what your machine cannot run, one subsystem per commit.** Each removal passes four gates:
   - a clean build;
   - `cuobjdump -sass` hashes per object and per function: kernels whose source did not change come out byte-identical,
     and inside a changed object only the removed functions may differ;
   - the CPU and GPU test suites;
   - greedy identity against the build before the removals (900 of 900 tokens, expert placement fixed).
6. **Choose each rounding by paired KLD against the reference, then by speed.** One switch per row, a per-chunk paired
   difference with a 95% interval (the KV table in `evals/2026-11-17/halocline-kv-kld/`).
7. **Add checks that KLD cannot see.** Needle retrieval at 36K and 137K and a tool-call check run on each serve.
8. **Publish every eval.** Command, raw output, logs and the script, next to the number.

## Using Halocline as a base for your own card

Do not run the shipped settings on another machine unchanged. They come from one 4090, one CPU, 64 GB of RAM and a SATA
pack disk. Fork the repo under your own name (`halocline-<arch>` keeps the lineage readable) and tune it to your parts.

**Another Ada card (4080, 4070, 4060 Ti, L4, L40).** The build works without code changes: every 40-series card is compute
capability 8.9, which is all the engine compiles. What changes is placement and speed:

- `--expert-cache auto` (or a slot count) and `--vram-reserve-mib`: how many expert blobs live in VRAM. Less VRAM means
  more experts in the page-locked RAM tier.
- `--prefill-cpu-share`: the share of streamed prompt experts the CPU computes (0.6 on the reference machine). An
  AVX-512 host can take more.
- `--ple-io`, `--ple-row-cache`, `--ple-inflight`: the n-gram table's reads. On a SATA pack disk the cold start is about
  0.5 GB/s; an NVMe helps the table and the cold start, not steady-state decode.
- `--kv`, `--max-context`, `--kv-resident`: the KV budget. int8 at 262K is 8.25 GiB.
- RAM is the tier that decides whether the model runs at all: the page-locked complement sizes to the pack and the
  budget, and with less RAM more experts stream from disk.

Then re-measure. Copy `SPECS.md`, run the speed floors on your machine (step 2), make your own BF16 base for the quality
gate (step 3; the published base scores only the model it was made from), and run the smoke. No number in this repo
carries to another card.

**Another GPU architecture.** That is kernel work, not a setting. Change the architecture check at the top of
`CMakeLists.txt`, then replace the kernels. They live in `src/kernels/cuda/`, `src/models/qwen4exp/kernels/` and
`src/prefill/`. Each has a parity test in `tests/` against a CPU or double-precision reference; keep the test and swap
the kernel behind the same host function. `ADA_PORT_NOTES.md` records what each Blackwell kernel became on Ada, which is
the template for a new architecture.

For another model architecture, fork per family (`halocline-<family>`) and write the adapter under `src/models/<family>/`.

## Sources

"Applied" means the source changed a design or a gate decision in this repo. "Read" means I read it and it changed
nothing; the reason is in the row. Every entry links to the paper or the code.

### Applied

| Source | What it reports | What Halocline took from it |
|---|---|---|
| [arXiv 2606.19558](https://arxiv.org/abs/2606.19558), "Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment" | KLD tracks benchmark scores across a full quant cohort and loses that link near the baseline ("silent zone") | KLD alone does not pass a release: needles at 36K and 137K and a tool-call check run beside it |
| [arXiv 2608.06564](https://arxiv.org/abs/2608.06564), "Which Decisions Low-Bit Quantization Breaks, and How to Predict Them" | low-bit quantization shrinks decision margins; whether to call a tool flips more often than which tool | the tool-call check stays in the smoke test |
| [arXiv 2606.00279](https://arxiv.org/abs/2606.00279), "Bit-Exact AI Inference Verification Without Performance Tradeoffs" | determinism and invariance are separate properties; atomic kernels are a nondeterminism source | Halocline's gates are SASS hashes per function and greedy identity, not a bit-exact runtime check |
| [Basalt precision docs](https://github.com/jesdga95/basalt/blob/main/docs/precision/quality.md) | the decode-path KLD gate and paired dKLD costs per rounding on the NVFP4 pack | the gate's method; our KV results compared against their published order |
| [Infernix](https://github.com/Wallawalla47/Infernix) (code, Apache-2.0) | TMA NVFP4 and FP8 GEMM tiles | the activation-scale layout and epilogue arithmetic of the dense FP4 and FP8 prompt products (`THIRD_PARTY_LICENSES`) |
| basalt_ada `ADA_PORT_NOTES.md` | the sm_89 replacements for clusters, TMA and the FP4 convert | the Ada kernels themselves |

### Read, not applied

| Source | Title | Why it changed nothing |
|---|---|---|
| [arXiv 2608.30320](https://arxiv.org/abs/2608.30320) | "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability" | the adapter came with the code; I have not checked it line by line against the report |
| [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) | "DFlash: Block Diffusion for Flash Speculative Decoding" | Strata's floors use DFlash2; Halocline runs the MTP drafter, and a DFlash2 A/B on Halocline is still open |
| [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) | "YaRN: Efficient Context Window Extension of Large Language Models" | the model runs 262,144 tokens natively |
| [arXiv 2610.15384](https://arxiv.org/abs/2610.15384) | "E8 Lattice Key Quantization for Low-Bit KV-Cache Compression" | int8 KV fits 262K on this card and measured closest to BF16 |
| [arXiv 2501.01005](https://arxiv.org/abs/2501.01005) | "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving" | its GDN kernels need sm_90 |
| [arXiv 2608.10103](https://arxiv.org/abs/2608.10103) | "Hand-Written PTX Tensor-Core GEMM Kernels: A Multi-Precision Study on NVIDIA L4" | background for reading the Ada kernels; no code taken from it |
| [arXiv 2604.18556](https://arxiv.org/abs/2604.18556) | "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling" | per-tensor quant allocation; Halocline keeps the Unsloth bytes unchanged by rule |
| [arXiv 2605.00649](https://arxiv.org/abs/2605.00649) | "Model Compression with Exact Budget Constraints via Riemannian Manifolds" | per-tensor quant allocation; same rule |
| [arXiv 2609.26173](https://arxiv.org/abs/2609.26173) | "Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity" | per-tensor quant allocation; same rule |
| [arXiv 2611.00486](https://arxiv.org/abs/2611.00486) | "Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units" | a lead; not dissected |
| [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) | "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU" | survey only |
| [arXiv 2508.15601](https://arxiv.org/abs/2508.15601) | "LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind" | survey only |
| [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) | "SGLang: Efficient Execution of Structured Language Model Programs" | survey only |
| [arXiv 2404.06654](https://arxiv.org/abs/2404.06654) | "RULER: What's the Real Context Size of Your Long-Context Language Models?" | the 36K and 137K needle checks already run in `lsw.py smoke --long`; no RULER harness added |

### Model and data sources

| Source | Revision | What Halocline uses |
|---|---|---|
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | the architecture (48 layers, 512 experts, 10 routed plus 1 shared, Gated DeltaNet and Qwen Sparse Attention, 262,144 context) and the 31 BF16 `mtp.*` tensors, 4.9 GiB, hash-checked per tensor by `tools/convert/mtp_fetch.py` |
| [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | `814db05f88b3b827ea7cfcab5344fd218de494d2` | the `UD-IQ4_XS` v3.1 shards, 93,672,359,738 bytes in four files, and the `BF16/` shards the KLD base was scored from |
| [jesdga/basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) | `a65bae55f703e9c6dab6e9f335146c532fb7dd95` | `flashnext-bf16-base.kld`, 12,664,728,820 bytes, sha256 `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9` |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `436f6f89e` | `llama-perplexity --kl-divergence-base`, the quantization formats and the reference quantizers |
| [wikitext-2](https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh) | | the reference text: `wiki.test.raw`, 100 chunks of 512 tokens |

Licences: the engine descends from MIT code (see `LIFTS.md`), Infernix is Apache-2.0, and the weights carry Qwen's
`qwen-community-1.0` licence. Read it before serving the model to anyone else.
