# How quality is measured

Halocline scores a model against a full-precision reference on the same text, through the decode path that generates
tokens. Both paths use this method. Their numbers: [Qwen3.8-Flash-Next](qwen38-flash-next.md) and
[Qwen3.8-27B](qwen38-27b.md).

## The gate

`halocline-bench --quality` runs the gate. `tools/ppl_gate.py` runs it offline against a built engine
([commands](#reproduce-the-reference)).

- **Text:** wikitext-2 in 512-token chunks. The first 256 tokens of each chunk are prefilled.
- **Scored rows:** 256 to 510, through the decode path in verify windows of `--quality-t` tokens (default 3). The
  window tokens are the reference text, so the drafter's output is not used.
- **Reference:** a llama.cpp KL-divergence base file (`--kl-divergence-base`).
- **KLD** is KL(base || test) over the full vocabulary, from the fp32 log-softmax, summed in double. Base terms at
  log-probability -16 or below are skipped, a tiny underestimate.
- **Reported:** PPL, mean KLD (the tables also give median, p99 and max) and top-1 agreement.
- **Length:** 20 chunks for a quick check. A change to numerics is scored on 100.
- **Determinism:** two runs of one build match row for row, with the placement pinned (see
  [greedy identity](#greedy-identity)).
- **Coverage:** the gate scores 512-token chunks, so it cannot see KV rounding that builds up deep in a long context.
  The depth bases (`tools/depth_kld`, KLD of a scored window after a 65K to 261K prefix) measure that;
  [`halocline-d8-depth`](../../evals/2026-11-17/halocline-d8-depth/) is the dense run, the 262K profile's rows are in
  [`halocline-long-context`](../../evals/2026-11-18/halocline-long-context/), and [testing](../testing.md) lists which
  depths are still open.

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Qwen3.8-Flash-Next | Unsloth's BF16 GGUF through `llama-perplexity` (llama.cpp `436f6f89e`), 100 chunks of 512, `-b 8192 -ub 8192` | 3.2749 (3.2733 on the scored rows) |
| Qwen3.8-27B | Unsloth's BF16 GGUF through `llama-perplexity` (llama.cpp `5676dfbe0`), wikitext-2, 100 chunks of 512, `-b 8192 -ub 8192` | 6.7437 (6.7411 on the scored rows) |

Each model has its own base because KLD is only comparable for the same model and tokenizer. The bases are published
as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on Hugging Face, with
checksums: pass one to `halocline-bench --quality --kld-base FILE` to score a build or a self-converted pack. The
dense base's hash is on [the 27B precision page](qwen38-27b.md).

A quantized base is not a reference: a Q8_0 GGUF adds its own rounding, which is why the bases here are BF16.

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy
identity.

## Greedy identity

Different expert placement rounds differently, so pin it first: `--adapt-swaps 0` turns the adaptive tier off, and a
fixed `--expert-cache N` pins the experts in VRAM. A dense pack has no experts and needs neither.

`tools/greedy_identity.py` runs two engine builds as `--serve` on the bench fixtures (conv, struct, prose) at
temperature 0 and compares every token:

```sh
python3 tools/greedy_identity.py REF_BIN NEW_BIN --model PACK.halo --tokens 300 \
    -- --spec 0 --expert-cache 7000 --prefill auto
```

It pins `--adapt-swaps 0` and `--max-context 8192` itself, and prints `GREEDY IDENTICAL` or `GREEDY DIFFERS` with the
first difference. The run from the pre-removal build (`901fd9c9`) to the published build read 900 of 900 tokens
([`halocline-greedy-identity-final`](../../evals/2026-11-17/halocline-greedy-identity-final/)).

Forced-window identity works on the gate's own rows: the same window width must give the same rows.

```sh
python3 tools/ppl_gate.py --model PACK.halo --base flashnext-bf16-base.kld --chunks 20 --t 3 --out DIR \
    -- --kv int8 --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR3/rows.tsv
```

## Reproduce the reference

Flash-Next base (the model does not fit in RAM and VRAM, so the experts stream from disk):

```sh
llama-perplexity -m Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-bf16-base.kld -c 512 -b 8192 -ub 8192 --chunks 100 -ngl 99 --n-cpu-moe 99 -t 16
```

The dense base's provenance and hash are on [the 27B precision page](qwen38-27b.md); the runs that score against it are
in [`halocline-dense-quality`](../../evals/2026-11-17/halocline-dense-quality/).
