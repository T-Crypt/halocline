# Qwen3.8-27B (dense)

> [!NOTE]
> Status 2026-11-17: the dense path does not run yet. The adapter and the shared dense stack exist; no 27B pack has been
> built or loaded, and the GGUF converter path is unwritten. The plan and its accept criteria are in
> [`docs/roadmap.md`](../roadmap.md#p1-the-dense-path-qwen38-27b-the-priority). This page replaces Basalt's NVFP4 pack
> page, which describes weights Halocline does not use.

Qwen3.8-27B is the dense half of the engine: a 27B hybrid with no experts and no router, standing in for Qwen4-27B until
its weights exist. It is the simpler model to hold close to BF16, and the path Qwen4-27B will use.

From the [model card and `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B):

| Property | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration`, model type `qwen3_5` |
| Layers | 64, three Gated DeltaNet layers to one gated attention layer |
| Hidden size | 5,120 |
| Attention | 24 query heads, 4 KV heads; GDN with 16 key and 48 value heads, head dimension 128 |
| FFN | dense SwiGLU, intermediate size 17,408 |
| Context | 262,144 native |
| Vocabulary | 248,320 |

## Weights

Only Unsloth's Unsloth Dynamic GGUFs at the revision on disk (`4af51b35`), per the maintainer's decision of 2026-11-17:

| File | Bytes | State |
|---|---|---|
| `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14,352,911,264 | downloaded, hash checked |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,659,407,264 | downloaded, hash checked |

The pack will repack these tensors losslessly, the same way the sparse pack treats quantized tensors: the codes and
scales are copied, and only the layout changes. No NVFP4 checkpoint, no requantization. Per-tensor hash equality with the
GGUF is the proof.

## Pack and serve

Not built. When it exists, the command is the same as any model:

```sh
bin/halo serve Qwen3.8-27B-UD-IQ4_XS.basalt --host 127.0.0.1 --port 8092 --max-context 262144 --kv int8
```

The BF16 reference for the quality gate is `qwen38-27b-bf16-base.kld`; the method and status are in
[precision/qwen38-27b.md](../precision/qwen38-27b.md).

## Why dense

- No router and no expert swaps, so there is less machinery that can move a token.
- Every parameter is read per token, so speed falls with quant size; quality rises.
- The model is small enough to keep its weights on the card, with the token embedding in page-locked RAM.

The comparison that settles whether the dense path is better than the sparse one for a task is task-level, not PPL:
perplexity is not comparable across two different models. See [`docs/testing.md`](../testing.md) and the SWE-bench plan
in the roadmap.
