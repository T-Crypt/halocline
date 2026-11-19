# Qwen3.8-27B (dense)

Qwen3.8-27B is the dense half of the engine: a 27B hybrid with no experts and no router, standing in for Qwen4-27B until
its weights exist. It is the simpler model to hold close to BF16, and the path Qwen4-27B will use.

The path runs since 2026-11-17 on Unsloth's v3.1 GGUF quants. The
[roadmap](../roadmap.md#p1-the-dense-path-qwen38-27b-the-priority) holds the state and every measured number: D1 is
decided; D2 to D4, the D6 state round trip and D8 are done; D5 and D7 have their NInfer runs, and the switch bar is not
met. The build-to-build greedy identity (D6) has not run.

D8 shipped on 2026-11-18: 262,144 tokens of context with MTP and vision on one 24 GB card, served as `--long-context`
(rotated 4-bit K and V, the drafter's int8 ring over 32,768 cells), and standing order 8 made it the dense default the
same day. K8V4 at 175,616 tokens is the lower-KLD lever, +0.00072 over int8 at 65,536
(`evals/2026-11-17/halocline-d8-depth`).

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

The served profile is UD-Q4_K_XL with rotated 4-bit K and V, MTP at `--spec 8`, vision on the CPU and 262,144 tokens of
context. `--long-context` sets the cache, the context and the drafter's int8 ring. The vision pack adds the projector
(`vision/mmproj`):

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 \
    --spec 8 --long-context --vision cpu
```

The K8V4 profile at 175,616 tokens is below. The BF16 reference for the quality gate is `qwen38-27b-bf16-base.kld`;
the method is in [precision/qwen38-27b.md](../precision/qwen38-27b.md).

## Two profiles

The dense path has two serving profiles. The default is `--long-context`: the model's full 262,144 tokens on rotated
4-bit K and V, at +0.0011 mean KLD over K8V4 at every measured depth (roadmap standing order 8). K8V4 at 175,616 tokens
is the lever for a run that wants the lower KLD and fits in the shorter window. Pick one per server start.

| | K8V4 (the lever) | `--long-context` (the default) |
|---|---|---|
| KV cache | `k8v4` (K int8, V rotated 4-bit) | `q4_0` (K and V rotated 4-bit) |
| Context | 175,616 tokens | 262,144 tokens (the model's native maximum) |
| Drafter cache | `--mtp-kv int8` | `--mtp-kv int8 --mtp-window 32768` (a ring over the last 32K cells; set by the flag) |
| Mean KLD vs BF16 at 65,536 / 140,288 / 174,080 tokens | 0.00490 / 0.00560 / 0.00654 | 0.00602 / 0.00676 / 0.00762 (+0.00112 / +0.00116 / +0.00108) |
| Reported as | | a lossy flag (the start warns, `/v1/status` lists it) |

Both need `--vision cpu` on a 24 GB card: at 262,144 tokens the plan fits with 53 MiB to spare and no room for the
vision encoder on the GPU. The default kernels are untouched by the second profile: its cache write and attention are
their own compiled instantiations, and the default's are instruction-identical to the release before it.

Two ready-made configs sit in `examples/`; copy one, point `model` at your pack, and start the server with it:

```sh
bin/halocline serve --config examples/qwen38-27b.json        # default: 262,144 tokens, rotated 4-bit K and V
bin/halocline serve --config examples/qwen38-27b-k8v4.json   # 175,616 tokens, K8V4: the lower KLD
```

Or add the flag to your own command line; any setting you give beside it wins, except `--kv`, which must be `q4_0` or
absent (the flag refuses another rather than silently replacing it):

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 \
    --spec 8 --long-context --vision cpu
```

To keep both on hand, run them as two entries of a model switcher (llama-swap, for example) on the same port with
different names, so a request picks the profile by model name and only one holds the card at a time.

## What is measured

- KLD against BF16 (D4): the measured table in the roadmap, from
  [`halocline-dense-quality`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-dense-quality).
  UD-Q4_K_XL is at llama.cpp parity on the same GGUF and 2.8 times better than UD-IQ4_XS.
- Speed (D5): decode and prefill fixtures in the same eval. Against NInfer with thinking off on both, K8V4 at 175,616:
  prose 9-10% under at short, 36K and 137K, JSON level at short and 8-9% over at depth
  (`evals/2026-11-18/halocline-vs-ninfer-nothink`). S1 (2026-11-18) took K8V4 decode past the NInfer floors: 36K prose
  150.6 to 158.8 and 137K prose 102.8 to 145.2 t/s against floors of 118 and 93 (`evals/2026-11-18/halocline-s1c-ab`).
  The 262K profile with W22's kernel reads 81.5 / 74.0 / 63.5 t/s on prose and 152 / 147 / 129 on JSON at short / 36K /
  137K (`evals/2026-11-18/halocline-w22`); prose stays about a tenth under NInfer there.
- Identity (D6): the state round trip passed on the real pack; the build-to-build greedy gate has not run.
- 262K fit (D8): probes in the same eval. Neither quant fits 262K at int8 or fp16 KV. Rotated 4-bit K and V with the
  drafter's int8 ring over 32,768 cells fits with 53 MiB to spare, and it is the default profile.
- K8V4 KV (D8 item 1):
  [`halocline-d8-kv`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-d8-kv). +0.00093 mean KLD
  over int8 at 36K (0.00676 against 0.00582), decode within 2%, prefill -4.6% at 36K, 175,629 tokens of context with
  `--mtp-kv int8`. The drafter's cache format leaves the scored output byte-identical.
- Rotated 4-bit K and V (D8 item 3): shipped on 2026-11-18 as `--long-context` (release ea5868de). Mean KLD 0.00602 /
  0.00676 / 0.00762 at 65,536 / 140,288 / 174,080, paired +0.00112 / +0.00116 / +0.00108 over K8V4, top-1 down 0.17 /
  0.15 / 0.33 points (`evals/2026-11-18/halocline-long-context`). A needle at 252,483 tokens was found. W22's staged Q4
  decode kernel is bitwise to the tiled one, keeps the depth KLD at +0.00000, and lifts 137K prose decode from 52.8 to
  63.5 t/s (`evals/2026-11-18/halocline-w22`).

For the K8V4 profile at 175,616 tokens (the 175,629-cell maximum rounded down to 256):

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 \
    --spec 8 --max-context 175616 --kv k8v4 --mtp-kv int8 --vision cpu
```

`examples/qwen38-27b-k8v4.json` is the same thing as a config. The depth gate passed (K8V4 over int8 +0.00072 at
65,536 and +0.00066 at 140,288; 0.00654 mean KLD at 174,080, [`halocline-d8-depth`](../../evals/2026-11-17/halocline-d8-depth/)),
and the load at 175,616 with MTP and vision served the full smoke through llama-swap: warm decode 113.3 to 145.9 t/s,
sampling, a tool call, the 36K needle (prompt 3,137 t/s, decode 102.2) and the 137K needle (prompt 2,295 t/s, decode
77.9).

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
2026-11-19: the state lines describe the 262K default and the depth results.
