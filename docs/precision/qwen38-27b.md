# Qwen3.8-27B precision

> [!NOTE]
> Status 2026-11-17: no dense run has been measured. This page holds the reference and the gate that will score it; the
> numbers appear when the pack loads. Basalt's NVFP4 numbers are not reproduced here because Halocline does not use that
> pack.

## The reference

| Item | Value |
|---|---|
| Base file | `qwen38-27b-bf16-base.kld`, 12,664,728,820 bytes, sha256 `9ffb3bd9ca44bd10b7fbf1f52b6691aad929778704299f084e65fdd19e482f30` |
| Made by | `llama-perplexity --kl-divergence-base`, llama.cpp `5676dfbe0`, wikitext-2 `wiki.test.raw`, 100 chunks of 512 tokens, `-b 8192 -ub 8192` |
| Base PPL | 6.7437 (6.7411 on the scored rows) |
| Hosted at | [jesdga/basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) at `a65bae55` |

The file is the same one the sparse path uses for its own model; each model has its own base because PPL and KLD are only
comparable for the same model and tokenizer.

## The gate

`halocline-bench --quality 100 --kld-base qwen38-27b-bf16-base.kld` scores the decode path and reports mean, median and p99
KLD, top-1 agreement, PPL, and a paired per-chunk difference with a 95% interval. The method is in
[quality.md](quality.md).

Both Unsloth quants will be scored (UD-IQ4_XS and UD-Q4_K_XL), each with the KV format that fits at 262,144 tokens:

| Config | Weights on card | KV at 262K | Headroom |
|---|---|---|---|
| UD-IQ4_XS + int8 KV | about 12 GiB | 8.25 GiB | tight |
| UD-IQ4_XS + k8v4 KV | about 12 GiB | 6.2 GiB | more |
| UD-Q4_K_XL + k8v4 KV | about 14.3 GiB | 6.2 GiB | tight |

The choice is whole-system KLD at depth first, speed second, as BUILD_SPEC §11.6 fixes it. Weight error and KV error
trade off, so neither is picked in advance.

## What has not run

- No load, no KLD, no greedy identity, no speed run.
- The baseline it must beat exists: NInfer-Qwen3.8-27B on the same card measured 112.4 to 118.3 tok/s warm decode,
  118.3 decode and 3,215 prompt at 36K, 93.0 decode and 2,155 prompt at 137K
  ([`evals/2026-11-16/halocline-p0-floors`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-16/halocline-p0-floors)).
  Those are speed floors. NInfer's quality on this base has not been scored.

## The comparison

The dense path's goal is lower KLD and better output than NInfer and llama.cpp on the same card and model. The rules:

- the same BF16 base for every arm;
- NInfer's weights differ because it requantizes, and the table says so rather than hiding it;
- speed is ABBA in one session, one engine on the card, power limit recorded;
- KLD near the baseline does not rank models, so the task suite in `docs/roadmap.md` runs beside it.

The table is filled when D2 to D5 of the roadmap land. Until then this page is a plan, not a result.

## KV formats and context (2026-11-17)

The K8V4 cache is in the dense GQA path (`--kv k8v4`, `src/kernels/cuda/gqa_attn.cu` and `dense_ops.cu`): K stays the
int8 form the dense path already had, V is stored in rotated q4_0 blocks (`include/halo/kernels/kv_q4_layout.hpp`, the
quantizer and block the sparse path's `q4_0` cache ships). The rotation is applied at the cache write and each head's
attention output is rotated back once, which is exact: a Hadamard transform commutes with attention over V.

| Format | Bytes a cell (one attention layer) | Max context on UD-Q4_K_XL, MTP and vision | Notes |
|---|---|---|---|
| `fp16` | 4,096 | 71,188 | the reference format |
| `int8` | 2,112 | 130,832 | the served profile (122,880) |
| `k8v4` | 1,632 | 164,094 | 184,694 once the MTP drafter's KV is a 32K ring (D8 item 2) |
| `q4_0` | 1,152 | 220,035 | 241,285 with `--mtp-kv int8`; 261,647 with the ring, 265,176 with the ring and int8 (2026-11-17) |

Numbers from `tests/kernels/dense_kv_planner_test.cpp` with the planner's own formula and the 7,597 MiB UD-Q4_K_XL
leaves free with vision packs (`evals/2026-11-17/halocline-dense-quality/fit/`). The 262,144-cell totals at int8
(11,830 MiB) and fp16 (19,766 MiB) are unchanged by the new format and are pinned by the same test. A cell is 4 KV heads
of 256 values plus the format's scales; K8V4 is 1,024 bytes of K codes, 32 bytes of K scales and 576 bytes of V blocks;
`q4_0` is 576 bytes of blocks a half, K and V.

No KLD has been scored for K8V4 on this model, and no decode or prefill number: those need the card. The gates it
carries are the repo's usual ones (CONTRIBUTING §"What every change must include"): the CPU quantizer test
(`kv_q4_layout_test`), the GPU parity test against a double-precision reference (`gqa_k8v4_parity`), and then the
100-chunk KLD table and the long-context windows of P0 item 3.

## Rotated 4-bit K and V (2026-11-17)

`--kv q4_0` stores K and V both in rotated q4_0 blocks (`KvQ4`, 1,152 B a cell, conversation cache format 4). K is
rotated after its rope at the cache write and quantized with the shared q4_0 block (`kv_q4_layout.hpp`), V as K8V4 has
it; Q is rotated before the attention call by the same 256-point transform (`H . H = 1`, so a logit is the unrotated dot
product), and each head's output is rotated back once. The prompt chunk reads it through the same tiled kernel as K8V4,
with K's tile expanded from blocks and one int32 accumulator a 32-dim block (86,016 B of shared memory at 4 warps).

At 262,144 cells the KV is 4,608 MiB, and the whole plan is 7,094 MiB with the MTP drafter's KV as the 32K ring of D8
item 2 (9 MiB over the card's 7,597 MiB with the 512 MiB reserve) or 7,032 MiB with `--mtp-kv int8` too (53 MiB to
spare). The maximum is 261,647 cells with the ring and 265,176 with the ring and int8. Both 4-bit formats refuse
`--kv-resident`, and `--mtp-kv q4_0` is refused: the drafter's cache stays fp16 or int8.

What has not run: the GPU parity (`gqa_k8v4_parity`'s Q4 arm, written and built), KLD for either 4-bit format, and
decode and prefill at any context. The CPU rotation and quantizer contract is tested (`kv_q4_layout_test`).

