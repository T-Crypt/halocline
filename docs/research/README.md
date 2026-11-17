# Research notes

Halocline started as a literature and measurement pass, before any engine code. These are the notes from that pass, kept
as written and scrubbed of lab paths and private hosts. They answer one question each, in the same shape:

- **Takeaway:** the answer in a paragraph.
- **Cited findings:** the evidence, each with a link or a file it came from.
- **Inferences:** what I concluded, marked as mine and not as a source's claim.
- **Gaps:** what the pass did not establish.

Read them as a working notebook, not as a paper. A claim in a note is only as good as the link beside it, and the notes
say where the link is missing. Paths that start with `~` name files on the machine the notes were written on; they are
kept as provenance, not as links a reader can follow.

## The notes

| Note | Question |
|---|---|
| `research_notes/sm89_quality_validation.md` | what "matches BF16" means, and which gates prove it |
| `research_notes/sm89_quantization_quality.md` | which weight formats preserve BF16 output on sm_89 |
| `research_notes/sm89_iq4xs_performance_quality_lane.md` | IQ4_XS as the engine's first quant: quality against speed |
| `research_notes/sm89_kernel_quality_bf16.md` | numerical quality of the Ada kernels against a BF16 reference |
| `research_notes/sm89_sparse_moe_quality.md` | MoE routing and expert offload, and whether placement touches the math |
| `research_notes/sm89_dense_quality.md` | the dense path, including the GDN hybrid |
| `research_notes/sm89_kv_context_quality.md` | KV formats and long context against BF16 |
| `research_notes/sm89_speculative_decoding_quality.md` | what a draft layer costs in output quality |
| `research_notes/sm89_memory_serving_quality.md` | memory pressure, offload and serving quality |
| `research_notes/sm89_qwen4_sparse_vs_dense_switch.md` | the Qwen4 family, and when sparse beats dense on one card |
| `reports/sm89 sparse dense engine quality.md` | the report that ties the notes together |

## Sources

Every arXiv id below was checked on arxiv.org on 2026-11-17, and the titles are quoted as arXiv prints them. Model
and data sources used by the engine are listed in [`RESEARCH.md`](../../RESEARCH.md).

| Paper | Title | Read for |
|---|---|---|
| [2606.19558](https://arxiv.org/abs/2606.19558) | Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment | whether KLD ranks near-baseline models, and what to run beside it |
| [2608.06564](https://arxiv.org/abs/2608.06564) | Which Decisions Low-Bit Quantization Breaks, and How to Predict Them | decision margins and tool-call stability under quantization |
| [2606.00279](https://arxiv.org/abs/2606.00279) | Bit-Exact AI Inference Verification Without Performance Tradeoffs | determinism, invariance, and where bit-exact checks fit |
| [2608.30320](https://arxiv.org/abs/2608.30320) | On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability | the model's architecture and training |
| [2602.06036](https://arxiv.org/abs/2602.06036) | DFlash: Block Diffusion for Flash Speculative Decoding | the drafter Strata runs, as a comparison for MTP |
| [2404.06654](https://arxiv.org/abs/2404.06654) | RULER: What's the Real Context Size of Your Long-Context Language Models? | long-context validation beyond needle retrieval |
| [2309.00071](https://arxiv.org/abs/2309.00071) | YaRN: Efficient Context Window Extension of Large Language Models | context extension past the native window |
| [2610.15384](https://arxiv.org/abs/2610.15384) | E8 Lattice Key Quantization for Low-Bit KV-Cache Compression | low-bit KV formats |
| [2501.01005](https://arxiv.org/abs/2501.01005) | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | attention and KV kernels |
| [2608.10103](https://arxiv.org/abs/2608.10103) | Hand-Written PTX Tensor-Core GEMM Kernels: A Multi-Precision Study on NVIDIA L4 | what sm_89 tensor cores can do, as background |
| [2604.18556](https://arxiv.org/abs/2604.18556) | GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling | per-tensor quant allocation |
| [2605.00649](https://arxiv.org/abs/2605.00649) | Model Compression with Exact Budget Constraints via Riemannian Manifolds | per-tensor quant allocation |
| [2609.26173](https://arxiv.org/abs/2609.26173) | Component Type, Not Reconstruction Error, Predicts Attention Quantization Sensitivity | which parts of a model tolerate quantization |
| [2611.00486](https://arxiv.org/abs/2611.00486) | Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units | microscaling on INT8 units |
| [2312.12456](https://arxiv.org/abs/2312.12456) | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | consumer-GPU expert offload |
| [2508.15601](https://arxiv.org/abs/2508.15601) | LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind | serving and mixed precision |
| [2312.07104](https://arxiv.org/abs/2312.07104) | SGLang: Efficient Execution of Structured Language Model Programs | serving design |

## What moved into the engine

Two of these changed a gate. The KLD limitation in 2606.19558 is why every serve also runs needle retrieval at 36K and
137K. The decision-margin result in 2608.06564 is why the smoke test keeps a tool-call check. The rest were read and
changed nothing; [`RESEARCH.md`](../../RESEARCH.md) gives the reason per row.
