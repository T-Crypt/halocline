# Evals

Every measurement in the README, `RESEARCH.md` and the commit messages has a directory here. Nothing in this repo is a
number without a run behind it.

## Layout

| Path | What it measures |
|---|---|
| `2026-11-16/halocline-p0-floors/` | the speed floors: Strata 0.2.10 with DFlash2, and NInfer-Qwen3.8-27B, on the same card and quant |
| `2026-11-17/halocline-first-load/` | the first Linux load of the sparse pack: conversion, placement, coherent output |
| `2026-11-17/halocline-greedy-identity/` | 900 of 900 greedy tokens identical across the scope removals, same expert placement |
| `2026-11-17/halocline-kv-kld/` | the KV format by KLD against BF16, then speed; and the draft window sweep. Raw `rows.tsv` per KV under `rows/` |
| `2026-11-17/halocline-sparse-serve/` | the first sparse serve without the drafter: smoke, needle at 36K and 137K |
| `2026-11-17/halocline-llama-swap/` | the same smoke through llama-swap with the drafter and int8 KV |

Each directory holds the README that explains the run, the commands, the raw JSON, the logs from the engine and the
script that drove it. `rows.tsv` files are the per-chunk gate output: one row per scored position, with the base and
test log-probabilities. They are large because the gate scores the full vocabulary.

## Re-running

The commands in each README name the binary, the pack and the flags. They were run from a release tree, not from the
build directory, so a rebuild does not move the numbers under a served model. The host is described in `SPECS.md`.
Numbers move with placement: pin `--expert-cache` and turn adaptive swaps off when comparing two builds.

The runs used the lab's GPU lease to take the card for the length of a job. Any equivalent way of keeping other work off
the card gives the same numbers; what matters is that nothing else holds VRAM or the PCIe link during a run.

## Reading a number

- A KLD number is against `flashnext-bf16-base.kld`, sha256
  `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9`, never against another quantized model.
- A speed number states the power limit and the prompt type. Decode on sampled text moves with the sampling; the
  structured fixture is the cleanest decode comparison.
- A greedy-identity number states the placement. Different expert placement rounds differently.
