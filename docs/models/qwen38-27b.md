# Qwen3.8-27B (dense)

Qwen3.8-27B is the dense half of the engine: a 27B hybrid with no experts and no router, standing in for Qwen4-27B until
its weights exist. It is the simpler model to hold close to BF16, and the path Qwen4-27B will use.

The path runs since 2026-11-17 on Unsloth's v3.1 GGUF quants. The
[roadmap](../roadmap.md#p1-the-dense-path-qwen38-27b-the-priority) holds the state and every measured number: D1 is
decided, D2 to D4 are done, D5 is open, D6 has the state round trip done with the build-to-build greedy identity open,
D7 is open, and D8 is open. D8 is 262,144 tokens of context with MTP and vision on one 24 GB card. The served profile
is 122,880 tokens at int8 KV; K8V4 KV with the drafter's int8 cache reaches 175,629, and 262,144 needs 4-bit K and V
(branch `d8-q4`: +0.00104 KLD over K8V4 at 36K).

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

Only Unsloth's Unsloth Dynamic GGUFs at the revision on disk (`4af51b35`), per the maintainer's decision of 2026-11-17.
The qwen35 branch of the converter ([`tools/convert/gguf_qwen35.py`](../../tools/convert/gguf_qwen35.py)) copies each
tensor as stored, and the per-tensor hash check is in
[`halocline-dense-load`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-dense-load).

| GGUF | Bytes | Pack |
|---|---|---|
| `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14,352,911,264 | `Qwen3.8-27B-UD-IQ4_XS.halo` |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,659,407,264 | `Qwen3.8-27B-UD-Q4_K_XL.halo` |

The engine rounds `ssm_alpha` and `ssm_beta` from Q8_0 to BF16 at load because the gating kernels read BF16. That is the
one rounding on the path: the pack holds the GGUF's bytes.

## Run

The served profile is UD-Q4_K_XL with int8 KV, MTP at `--spec 8`, vision on the CPU and 122,880 tokens of context. The
vision pack adds the projector (`vision/mmproj`):

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 \
    --spec 8 --max-context 122880 --kv int8 --vision cpu
```

The BF16 reference for the quality gate is `qwen38-27b-bf16-base.kld`; the method is in
[precision/qwen38-27b.md](../precision/qwen38-27b.md).

## What is measured

- KLD against BF16 (D4): the measured table in the roadmap, from
  [`halocline-dense-quality`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-dense-quality).
  UD-Q4_K_XL is at llama.cpp parity on the same GGUF and 2.8 times better than UD-IQ4_XS.
- Speed (D5): decode and prefill fixtures in the same eval. The like-for-like NInfer comparison is open (D5, D7).
- Identity (D6): the state round trip passed on the real pack; the build-to-build greedy gate has not run.
- 262K fit (D8): probes in the same eval. Neither quant fits 262K at int8 or fp16 KV.
- K8V4 KV (D8 item 1):
  [`halocline-d8-kv`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-d8-kv). +0.00093 mean KLD
  over int8 at 36K (0.00676 against 0.00582), decode within 2%, prefill -4.6% at 36K, 175,629 tokens of context with
  `--mtp-kv int8`. The drafter's cache format leaves the scored output byte-identical.
- Rotated 4-bit K and V (D8 item 3, branch `d8-q4`): 0.00779 mean KLD at 36K, +0.00104 over K8V4; 262,144 tokens fit
  with `--mtp-kv int8 --mtp-window 32768`. Same eval, arm `q4`.

For long context, the planner's setting (175,616 = the 175,629-cell maximum rounded down to 256; not load-probed yet) is:

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 \
    --spec 8 --max-context 175616 --kv k8v4 --mtp-kv int8 --vision cpu
```

It is not the served default until a load probe at that size and the depth gate (KLD at 65K to 174K tokens) have passed.

## Why dense

- No router and no expert swaps, so there is less machinery that can move a token.
- Every parameter is read per token, so speed falls with quant size; quality rises.
- The model is small enough to keep its weights on the card, with the token embedding in page-locked RAM.

The comparison that settles whether the dense path is better than the sparse one for a task is task-level, not PPL:
perplexity is not comparable across two different models. See [`docs/testing.md`](../testing.md) and the SWE-bench plan
in the roadmap.

## History

2026-11-17: the first version of this page said the dense path did not run and that no pack had been built. The
converter, both packs, the first load and the quality gate landed the same day; this page now describes that state.
