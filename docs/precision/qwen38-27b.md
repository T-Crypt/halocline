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

## What has run (2026-11-17)

- The D4 gate on both quants ([`evals/2026-11-17/halocline-dense-quality`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-dense-quality/)):
  UD-Q4_K_XL with int8 KV scores 0.00582 mean KLD and 96.67% top-1 against BF16, within llama.cpp's own error bar on
  the same GGUF (0.00573 with f16 KV). UD-IQ4_XS scores 0.0162, 2.8 times worse, so UD-Q4_K_XL is the dense baseline.
- K8V4 KV against int8 on UD-Q4_K_XL ([`evals/2026-11-17/halocline-d8-kv`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-d8-kv/)):
  below.
- Not yet: KLD at depth (the BF16 depth bases exist; the runs are `evals/2026-11-17/halocline-d8-depth`, pending), a
  quality score for NInfer on this base, and the task suite.
- The speed floor to beat: NInfer-Qwen3.8-27B on the same card measured 112.4 to 118.3 tok/s warm decode,
  118.3 decode and 3,215 prompt at 36K, 93.0 decode and 2,155 prompt at 137K
  ([`evals/2026-11-16/halocline-p0-floors`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-16/halocline-p0-floors/)).

## The comparison

The dense path's goal is lower KLD and better output than NInfer and llama.cpp on the same card and model. The rules:

- the same BF16 base for every arm;
- NInfer's weights differ because it requantizes, and the table says so rather than hiding it;
- speed is ABBA in one session, one engine on the card, power limit recorded;
- KLD near the baseline does not rank models, so the task suite in `docs/roadmap.md` runs beside it.

D2 to D4 have landed; the NInfer quality arm (D7) has not run, so the comparison table is not filled yet.

## KV formats and context (2026-11-17)

The K8V4 cache is in the dense GQA path (`--kv k8v4`, `src/kernels/cuda/gqa_attn.cu` and `dense_ops.cu`): K stays the
int8 form the dense path already had, V is stored in rotated q4_0 blocks (`include/halo/kernels/kv_q4_layout.hpp`, the
quantizer and block the sparse path's `q4_0` cache ships). The rotation is applied at the cache write and each head's
attention output is rotated back once, which is exact: a Hadamard transform commutes with attention over V.

| Format | Bytes a cell (one attention layer) | Max context on UD-Q4_K_XL, MTP and vision | Mean KLD at 36K (100 chunks) |
|---|---|---|---|
| `fp16` | 4,096 | 71,188 | not scored (the reference format) |
| `int8` | 2,112 | 130,832; 138,061 with `--mtp-kv int8` | 0.00582, top-1 96.67% |
| `k8v4` | 1,632 | 164,094; **175,629 with `--mtp-kv int8`**; 187,183 adding the 32K drafter ring | 0.00676, top-1 96.39%: +0.00093 [+0.00074, +0.00113] paired |
| `q4_0` | 1,152 | 220,035; 241,285 with `--mtp-kv int8`; **265,176 adding the 32K ring** (262,144 fits) | 0.00779, top-1 96.07%: +0.00197 [+0.00167, +0.00228] paired |

Context numbers from `tests/kernels/dense_kv_planner_test.cpp` with the planner's own formula and the 7,597 MiB
UD-Q4_K_XL leaves free with vision packs; the q4_0 row from the same test on `d8-q4` (`evals/2026-11-17/halocline-dense-quality/fit/`). The 262,144-cell totals at
int8 (11,830 MiB) and fp16 (19,766 MiB) are pinned by the same test. A cell is 4 KV heads of 256 values plus the
format's scales; K8V4 is 1,024 bytes of K codes, 32 bytes of K scales and 576 bytes of V blocks. KLD from
[`evals/2026-11-17/halocline-d8-kv`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-d8-kv/).

- **K8V4's cost is small and measurable**: +16% mean KLD over int8 at 36K, 0.28 top-1 points, for 27% more context at
  the same drafter settings. Whether that cost grows with depth is what the depth bases decide.
- **The drafter's cache is free**: `--mtp-kv int8` gives byte-identical scored rows to `--mtp-kv fp16`, because the
  verify pass decides every token. It buys 7,000 to 11,500 cells and is the recommended setting with either format.
- Speed: decode within 2% of int8; prefill -2.1% at 16K and -4.6% at 36K (8-warp kernel, ABBA).
- 262,144 cells need 4-bit K too: rotated q4_0 K and V (`--kv q4_0`, 1,152 bytes a cell, branch `d8-q4`) costs
  +0.00104 over K8V4 at 36K (paired), so each 4-bit half adds about 0.001; decode on par; prefill -15% at 36K (its
  prompt kernel runs 4 warps only). 262,144 cells fit only with both drafter levers (`--mtp-kv int8 --mtp-window 32768`).
  Whether these gaps grow with depth is the depth gate's question (roadmap D8 item 3).

## KV formats at depth (2026-11-18)

The depth gate answers that question for K8V4: its cost does not grow. Each depth's base scores 1,024 rows after a
prefix of that many tokens, 16 chunks ([`halocline-d8-depth`](../../evals/2026-11-17/halocline-d8-depth/)); compare
formats within a depth, since each depth scores different text.

| Depth | int8 | K8V4 (paired vs int8) | rotated 4-bit (paired vs int8) |
|---|---|---|---|
| 36,864 (the 100-chunk gate, [`halocline-d8-kv`](../../evals/2026-11-17/halocline-d8-kv/)) | 0.00582 | +0.00093 [+0.00074, +0.00113] | +0.00197 [+0.00167, +0.00228] |
| 65,536 | 0.00418 | +0.00072 [+0.00050, +0.00095] | +0.00185 [+0.00135, +0.00234] |
| 140,288 | 0.00494 | +0.00066 [+0.00050, +0.00082] | not run |
| 174,080 | does not fit | 0.00654 alone | not run |

Rotated 4-bit K and V at 261,119 has not run: that protocol needs 262,152 cells, past the model's 262,144.
