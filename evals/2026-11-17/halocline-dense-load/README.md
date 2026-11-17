# Halocline: first dense load, Qwen3.8-27B from Unsloth GGUFs (D3), 2026-11-17 (the workstation, RTX 4090, ticket the board)

The first time the dense path (`src/models/qwen35` on `src/dense`) loaded a lossless GGUF repack and generated.

- Packs (D2, `ffb25a21`, both `gguf_qwen35.py verify` 866/866 LOSSLESS):
  `/data/models/halocline/packs/Qwen3.8-27B-UD-IQ4_XS.basalt` (13.385 GiB, sha256 `0a65f66d…`) and
  `Qwen3.8-27B-UD-Q4_K_XL.basalt` (sha256 `7fad5e7f…`), from `/mnt/models/Ollama/unsloth/Qwen3.8-27B-GGUF/`.
- Build: `~/wt/halocline-import` `scope-wave1` at `ffb25a21` plus the uncommitted engine change below, `build-89h`.
- Run: `berth run --as halocline-dense-load --priority foreground --pinned -- bin/halo --model PACK --tokens-file
  prompt.ids --max-new 96 --max-context 8192` (greedy; drafter defaults). Prompt: chat template, thinking off,
  "Explain the difference between a B-tree and an LSM tree in three sentences." `tok.py` is the tokenizer wrapper.

| Pack | On the GPU | Embedding (page-locked) | Prefill | Decode, 96 tokens | Output |
|---|---|---|---|---|---|
| UD-IQ4_XS | 12.87 GiB, 817 tensors | 0.51 GiB (Q3_K as stored) | 27 tokens in 41 ms | 1,853 ms | coherent, correct (`iq4xs.text.txt`) |
| UD-Q4_K_XL | 15.79 GiB, 817 tensors | 0.67 GiB (Q4_K as stored) | 27 tokens in 41 ms | 2,459 ms | coherent, correct (`q4kxl.text.txt`) |

No CUDA error, no load warning. D3 accept: met for both quants. The embedding is smaller than the spec's 2.4 GiB
because it stays in the GGUF's type and the gather kernel dequantizes each row.

## The engine change that made it load

Before this, the dense stack ran only Q8_0, NVFP4_PLANES and F8 matrices and wanted a BF16 embedding and BF16 GDN
alpha/beta. An Unsloth Dynamic GGUF mixes 12 llama.cpp types per tensor. The change:

- `meta.hpp`: every GGUF type MMQ has an instance for (Q8_0, Q5_0/1, Q2_K-Q6_K, the i-quants) is a dense matrix, rows
  in whole blocks; the adapter also asks `mmq::fits` per matrix. `linear_types_test` covers the mix and the refusals.
- The embedding keeps its stored type (`Weights::embd_type`, `embd_row_bytes`); `iq_embed_rows` already read them.
- Prompt chunks: `gemm` readies the activations again when a matrix's type differs from the one `chunk_act` last
  readied for (attention k and v, GDN qkv and z can differ in a UD GGUF; MMQ's q8_1 layout is per type).
- **One rounding, at load, not in the pack:** the GGUF's Q8_0 `ssm_alpha`/`ssm_beta` (2 x 48 x 5,120 values a GDN layer)
  go up as BF16 (ggml's fp32 dequant, nearest-even), because the GDN gating kernels read BF16. The pack stays byte-exact.
  An exact path (alpha/beta as Q8_0 MMQ products, gating after) is open; D4 against llama.cpp on the same GGUF decides
  how much it matters.
- The weight planner counts those two tensors at BF16 size (the first load ran out of its block by 22 MB on `mtp.*`).
- Also fixed: `tests/program/community_row_test.cpp` did not compile since `f247813c` (positional `CommunityGpu`
  initializers after `cuda_cores`/`compute_cap` were added); the "build clean" gate on `cf499585` was not true for
  `build-89h`.

Gates on the change: build clean, CPU 86/86 (`TMPDIR=~/wt/halocline-tools CUDA_VISIBLE_DEVICES= ctest -L cpu`).
GPU suite not yet run on it.
