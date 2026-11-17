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

`halo-bench --quality 100 --kld-base qwen38-27b-bf16-base.kld` scores the decode path and reports mean, median and p99
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
