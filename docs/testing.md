# Testing and quality gates

What runs today, what each gate proves, and what is still missing. The method pages are `docs/precision/quality.md` and
`docs/research/FINDINGS.md`; this page is the checklist.

## What runs on every engine change

| Gate | What it proves | How it runs | Current result |
|---|---|---|---|
| Build | the tree compiles for sm_89 with no warnings | `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && ninja -C build` | 728 targets, clean |
| SASS per object and per function | device code the change does not target is byte-identical | `tools/sass_gate.py`, hashes in `gates/` | 101 objects, 2,724 functions |
| CPU suite | host logic, formats, tokenizers, planning | `ctest -L cpu -j 16` | 86 of 86 |
| GPU suite | kernels and their parity tests against CPU or double references | `ctest -L gpu -j 1` under a GPU lease | 122 of 122; with `HALO_QWEN35_PACK` set the 27B round trip runs too |
| Python server suite | request parsing, streaming, tools, config, images | `python -m unittest discover -s serve/tests -t .` | 1,079 OK, 3 skips |
| KLD gate, sparse | output distribution against the frozen BF16 base | `halocline-bench --quality 100 --kld-base ...` | mean 0.08195, top-1 90.45% |
| KLD gate, dense | the same, against `qwen38-27b-bf16-base.kld` | the same | UD-Q4_K_XL 0.00582 / 96.67%; UD-IQ4_XS 0.01618 / 94.31% (int8 KV) |
| Dense state round trip | a saved session restores to the same bytes and tokens | `qwen35_checkpoint_round_trip` with `HALO_QWEN35_PACK` | passes on the UD-IQ4_XS pack, plain and with the drafter |
| Greedy identity | a change that must not alter output does not | `tools/greedy_identity.py REF NEW --model PACK` | 900 of 900 tokens, pre-removal build to published build |
| Smoke | the served model answers, streams, calls a tool, and retrieves at 36K and 137K | `lsw.py smoke --long` through llama-swap | all checks pass |

The 7 GPU skips are fixed by the hardware and the packs on disk: four are sm_120-only kernels, one needs two GPUs, one
needs a 27B pack, and one is the end-to-end generate/prompt test. `logs/ctest-gpu.sm89.log` lists them by name.

## What the gates do not cover yet

Each row is a gate that is missing or has been run once. "Blocks" says what cannot be claimed until it runs.

| Gate | Why it matters | Acceptance | Blocks |
|---|---|---|---|
| Dense 262K, identity, speed | load, KLD and state passed on 2026-11-17; 262K does not fit with fp16/int8 KV (roadmap D8), build-to-build identity and the like-for-like NInfer smoke have not run | a 262K load with MTP and vision, greedy identity across two builds, `lsw.py smoke --long` above the NInfer floors | the claim that dense replaces NInfer |
| Varying-context KLD | the gate scores 512-token chunks with a 256-token prefill, so KV error at depth is invisible | KLD of scored windows at 64K, 137K and 200K of context, per KV format, against the same base | the claim that int8 KV holds up at 262K, beyond needle retrieval |
| Sampling distribution | greedy identity says nothing about the sampler under temperature | seeded sampling, N runs per prompt, top-1 and top-10 frequency within the run-to-run interval of the reference build | "sampling is unchanged" |
| Task suite (SWE-bench) | KLD does not predict task outcomes near the baseline | fixed subset, fixed scaffold, resolve rate with an interval, recorded per config | any "as good as BF16 on tasks" claim |
| Broader tool-call set | the paper behind the tool check found that whether-to-call flips before which-tool | 20 or more prompts, whether-to-call accuracy against the BF16 reference path | "tool calls are preserved" beyond one example |
| RULER-style long context | one needle at two depths is a retrieval check, not a context-quality check | multi-needle, multi-hop and aggregation tasks at 36K, 137K, 200K | long-context quality claims |
| Strata KLD | the speed table compares two engines with different KV and drafters | the same gate on Strata 0.2.10 with its own reference | any quality comparison between engines |
| Container image | the Dockerfile is untested | build, serve, one smoke inside the image | container support |
| Concurrency and cancellation | one slot is the supported mode; cancellation and reconnect paths are untested under load | scripted concurrent clients, aborts mid-stream, no stuck leases | multi-client serving |

## BF16 comparison: what is and is not possible here

The reference is a file, not a running model. `flashnext-bf16-base.kld` holds the BF16 next-token distribution for 100
chunks of wikitext-2, made with llama.cpp. Scoring against it is cheap and repeatable.

A like-for-like BF16 serve of Qwen3.8-Flash-Next is not possible on this card: at BF16 the model needs about 250 GB
against 24 GB of VRAM and 64 GB of RAM. The comparison therefore stays distributional for the sparse model. That is why
the canaries exist: KLD cannot see everything.

For the dense 27B the same limit applies in milder form. The BF16 GGUF is about 54 GB, more than the card holds, so the
dense gate is also a KLD comparison against `qwen38-27b-bf16-base.kld`, plus decode speed against the floor. Nobody has
run it.

## SWE-bench

SWE-bench measures issue resolution on real repositories. It is the first task-level gate worth adding, because it
exercises long context, tool use and code editing together, and because a 4-bit MoE is exactly the case where
distributional distance can hide task regressions.

The plan, with the commands and the blocker list, is in `docs/roadmap.md#p2-task-level-eval-swe-bench`. The short version: use a fixed
subset (SWE-bench Lite first), one scaffold, temperature 0, a pinned container set, and report the resolve rate with a
confidence interval next to the KLD numbers for the same pack and KV format.

## What a release needs

A release is gated when every row in the first table passes on the release commit, the dense path has its own row of
those gates, and no item in the second table is marked as blocking for the claims in the README. Today the sparse path
meets the first table and the second table's rows are open work; the dense path meets neither.
