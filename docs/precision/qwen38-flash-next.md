# Qwen3.8-Flash-Next precision

What the sparse path computes in, where it rounds, and what that costs against BF16. The method is shared:
[how quality is measured](quality.md). The measured tables are in the [README](../../README.md#results); this page
explains where the roundings come from.

## The reference

| Item | Value |
|---|---|
| Base file | `flashnext-bf16-base.kld`, 12,664,728,820 bytes, sha256 `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9` |
| Made by | `llama-perplexity --kl-divergence-base`, llama.cpp `436f6f89e`, wikitext-2 `wiki.test.raw`, 100 chunks of 512 tokens, `-b 8192 -ub 8192` |
| Base PPL | 3.2749 (3.2733 on the scored rows) |
| Hosted at | [jesdga/basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) at `a65bae55` |

The gate scores 256 to 510 rows of each chunk through the decode path, in verify windows of 3 tokens. The window tokens
are reference text, so the drafter's output never enters the score. Reported: mean, median and p99 KLD, top-1 agreement,
PPL, and a paired per-chunk difference with a 95% interval.

## Weights

The pack keeps Unsloth's mixed-encoding 4-bit file: every quantized tensor's codes and scales are copied, not
re-quantized. The file is 93,672,359,738 bytes; IQ4_XS is one of several encodings in it, not the whole file.

Two roundings enter at conversion:

- **The `hc_*` cast.** 195 tensors are Q8_0 in the GGUF and BF16 in the pack (`--compat-bf16`). The engine reads BF16
  hc tensors before it makes its own 8-bit `hc8` copies. The cast is interim, and the KLD numbers include it.
- **The draft layer.** 31 BF16 `mtp.*` tensors from the model repository, hash-checked per tensor, packed as the
  `mtp/dense` and `mtp/experts` sections (0.77 GiB). A drafted token is verified against the target model, so the drafter
  cannot change output.

## Decode

- Experts run as int8 MMQ: the IQ4_XS codes decode inside the dot product, activations quantize to q8_1, and the
  accumulation is exact int32 per block with fp32 scales.
- Experts that do not fit the card run on the CPU over AVX2, from the page-locked RAM complement. The CPU computes the
  same values with its own kernels; placement changes latency, not text.
- The KV cache holds int8 codes with a scale per block. The paired gate put int8 ahead of k8v4 and q4_0 and level with
  fp16, and fp16 halves the 131K prefill, so int8 is the setting.
- The 36 Gated DeltaNet layers keep a fixed-size recurrent state in fp32; the 12 Qwen Sparse Attention layers keep the KV
  cache and read a fixed 2,048-token budget through a small indexer.

## Prefill and the Ada kernels

Prefill routes the same experts through MMQ with longer chunks, and gives the CPU 0.6 of the streamed experts. Ada has
int8, BF16 and FP16 `mma.sync`, `cp.async` and 99 KB opt-in shared memory. It has no FP4 tensor cores, no thread-block
clusters, no TMA and no programmatic dependent launch, so the Blackwell kernels were removed and their Ada replacements
live in `src/kernels/cuda/`, `src/models/qwen4exp/kernels/` and `src/prefill/`. `ADA_PORT_NOTES.md` records what replaced
what and the development A/Bs.

One kernel fact worth stating: FP8 W8A8 is not exact on Ada. The E4M3 `mma.sync` form loses too many bits in its sums at
K 6144 for the parity test, so the FP8 prompt product decodes to FP16 and runs FP16 `mma.sync` with fp32 accumulation.

## Limits

- The gate scores 512-token chunks, so it cannot see KV rounding that builds up deep in a long context. Needle retrieval
  at 36K and 137K is the current guard; KLD at depth is open work in [`docs/testing.md`](../testing.md).
- The weight quant's share of the KLD is not isolated. The four KV formats span 0.0065 KLD and the whole build reads
  0.0820.
- The `hc_*` BF16 cast is interim.
