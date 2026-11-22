# `branches/halo-inkling/`

Inkling 2 Mini (Thinking Machines, released 2026-11-20, Apache 2.0) on Halocline's own forward. **Experimental.** This
branch is not supported the way `main` is: it is published so others can build and test it on their own Ada card. The
numbers below are the short tests run so far; longer tests land here as they finish. `main` never writes in this folder.

## The model

A looped MoE transformer: 24 stored layers, 48 layer visits (layers 0-5 once, the core 6-17 three times with shared
weights, 18-23 once), 26.2B parameters with 3.4B active per token. Each visit keeps its own K/V and short-convolution
state. Attention is 20 sliding-window layers (1,024) and 4 full layers, with a relative position bias, a gate and a
sink logit; 24 query heads over 8 K/V heads of 128. Block attention residuals mix earlier blocks per token, and from the
second pass on every core layer reads a message from the previous pass through a rank-64 gate. 128 routed experts,
top 8, in NVFP4 (layers 0-19) and MXFP8 (layers 20-23) as trained. Vocabulary 200,058 tokens, GPT-4o pre-tokenizer.

## What this branch adds

| Path | What |
|---|---|
| `src/models/inkling2/model.cu` | the weights as the GGUF stores them, the 48-visit forward, the session (prefill chunks, decode windows, a CUDA graph for one-row decode, logit bias, log-probabilities, running-state checkpoints) |
| `src/models/inkling2/kernels/ink_ops.*` | the model's own kernels: relative-bias attention with a sink (split-K, any window), depth mix, loop controllers, short convolutions, MoE routing |
| `src/models/inkling2/adapter.cpp` | registration, tokenizer, special tokens, tool-call form |
| `src/program/tokenizer/pretok_gpt4o.cpp` | the GPT-4o pre-tokenizer (300 fixed cases and 35,000 random ones agree with Python's `regex`) |
| `tools/convert/gguf_inkling2.py` | the `.halo` pack from Unsloth's GGUF: every tensor copied as stored, the DSpark drafter carried as a section |
| `serve/models/inkling.py` | the chat dialect: the model's template, thinking, and its JSON tool calls |
| `tests/models/inkling2/` | `inkling2_greedy` (greedy ids) and `inkling2_kld` (KLD against a llama.cpp base) |

Expert and Q8_0 products run on the engine's MMQ and native mmvq; nothing in the shared engine changed except the
pre-tokenizer and the model list.

## Build and run

```sh
cmake -S . -B build-89 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DHALO_MODELS=inkling2
ninja -C build-89 halocline inkling2_greedy inkling2_kld
# the pack: Unsloth's root-layout GGUF (not Mainline_Support/), unsloth/Inkling-2-Mini-GGUF rev ae9327c5
python3 tools/convert/convert.py --gguf Inkling-2-Mini-UD-Q4_K_XL.gguf --no-vision \
    --dspark dspark/dspark-Inkling-2-Mini-BF16.gguf --out Inkling-2-Mini-UD-Q4_K_XL.halo
bin/halocline serve Inkling-2-Mini-UD-Q4_K_XL.halo --host 127.0.0.1 --port 8097 --alias Inkling-2-Mini \
    --max-context 262144 --kv int8 --no-grammar
```

`--no-grammar`: the shared tool grammar has no form for JSON arguments yet. The dialect still parses the model's calls.

## Short tests (RTX 4090, 2026-11-21)

Reference: llama.cpp from Unsloth's `b12588-mix-c0c3de8` release source, on the same GGUF.

| Test | Result |
|---|---|
| Greedy, a 26-token chat prompt | identical to llama.cpp, 24 of 24 tokens and through the end of the answer ([`first-light`](evals/2026-11-21/first-light/)) |
| KLD against UD-Q8_K_XL (the released NVFP4 experts with the rest in BF16), wikitext-2, 100 x 512 | Halocline, BF16 K/V: **0.003247 +- 0.000138**, top-1 97.65%, PPL 7.0747 (base 7.0623). llama.cpp, same weights, f16 K/V: 0.003199 +- 0.000102, top-1 97.74% ([`kld-short`](evals/2026-11-21/kld-short/)) |
| Decode | 84.5 t/s after a short prompt, 74.2 t/s after 4K |
| Prefill | 2,053 t/s at 4,095 tokens |
| Server | chat, a coding answer, a tool call and its result, prefix reuse (169 of 191 tokens cached on the second turn) |
| VRAM | 18.7 GB at 32K context, 21.8 GB at 131K (BF16 K/V); int8 K/V at 262,144: 21.6 GB |
| K/V formats in the engine (`--kv`, full-attention caches only), KLD 100 x 512 | bf16 0.003204, int8 **0.003274**, k8v4 0.003930, rotated q4 0.005406: the same as llama.cpp's split (0.00320 / 0.00327 / 0.00386 / 0.00541) |
| Long context, int8 K/V, served | a fact placed at 30% of 191,486 tokens of wikitext is retrieved: 129 s for the whole request with the prompt attention on the tensor cores (325 s before, prefill 592 t/s); decode 55.5 t/s at that depth (70.2 t/s at 29K) |
| Prefill at a 64K prompt | 1,993 t/s (1,136 before the tensor-core attention and the cuBLAS loop controllers) |

Quant: UD-Q4_K_XL. Every Unsloth 4-bit file keeps the released NVFP4 experts byte for byte; in llama.cpp UD-Q4_K_XL scores
0.00320 and UD-IQ4_XS 0.00397 against UD-Q8_K_XL.

K/V, from llama.cpp ([`llamacpp-kv`](evals/2026-11-21/llamacpp-kv/), with a 12-line patch that keeps the sliding-window
caches at f16 and quantizes only the 8 full-attention ones):

| K/V | KLD at 512 | KLD at a 65,536 prefix |
|---|---|---|
| f16 | 0.00320 | 0.00258 |
| q8_0 everywhere | 0.00327 | 0.00274 |
| K q8_0 / V q4_0, full caches only | 0.00386 | 0.00335 |
| rotated q4_0, full caches only | 0.00541 | 0.00479 |
| rotated q4_0, every cache | 0.01005 | - |

Quantizing the 40 sliding-window caches costs as much as the 8 full ones, and they are only 1,024 positions long.

## Not done yet

- KLD at depth (65K, 262K) for the K/V formats; `--kv q4` decodes far too slowly at a 246K prefix (0.3 t/s), a bug
- prefill: the MoE grouped GEMM (~107 ms a 512-row chunk) and the depth mixes are the next costs
- the DSpark drafter, a JSON tool grammar, images
- faster decode: the bandwidth floor is about 4 GB a token, near 225 t/s on a 4090
