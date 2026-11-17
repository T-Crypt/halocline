# Roadmap and open work

The engine has two paths. The sparse path runs Qwen3.8-Flash-Next today. The dense path is for Qwen3.8-27B now and
Qwen4-27B later. Its code exists; no real pack has been through it.

The dense goal, set by the maintainer on 2026-11-17: **Qwen3.8-27B on Unsloth's v3.1 GGUF quants only, held to the same
gates as the sparse path, with lower KLD to BF16 and better output quality than NInfer and llama.cpp on the same card.**

Status:

| Area | State |
|---|---|
| Sparse path (Flash-Next) | runs, gated, served through llama-swap |
| Dense path (27B) | adapter, shared dense stack and session state written; converter reads NVFP4 checkpoints only; no GGUF pack path; no pack built or loaded; both 27B tests skip without one |
| Dense vs NInfer / llama.cpp | not measured; no 27B run exists |
| Qwen4-Flash / Qwen4-27B | target only; weights are not public |
| Task-level eval | not started |
| Sampling, long-context KLD, Strata KLD | not measured |

## P1: the dense path (Qwen3.8-27B), the priority

The model is a dense hybrid: 64 layers, hidden size 5,120, three Gated DeltaNet layers to one gated attention layer,
SwiGLU FFN in each, 248,320 vocabulary, 262,144 native context. No experts, no router, no expert tiering. That is why
it is the quality path: there is less machinery between the weights and the logits than the sparse model has.

What exists: `src/models/qwen35/adapter.cpp` (registers the architecture, reads the pack, builds sessions),
the shared dense stack in `src/dense/`, `tools/convert/checkpoint_qwen35.py` (an NVFP4 checkpoint to a pack), and
`qwen35_checkpoint_round_trip` and `qwen35_linear_types_test`, which skip without a 27B pack.

### D1 (decided): Unsloth v3.1 GGUFs only

The dense path builds from Unsloth's Unsloth Dynamic GGUFs at the revision on disk (`4af51b35`): UD-IQ4_XS (14.35 GB)
and UD-Q4_K_XL (17.66 GB). No NVFP4 checkpoint, no QUASAR or NVIDIA requantization, no plain absmax requant that throws
away Unsloth's imatrix allocation. The NVFP4 converter stays in the tree as a tool and is not the path for any claim.

This closes BUILD_SPEC §11.1 in the direction the spec already chose: repack, do not requantize. The work it implies is
the qwen35 branch of the GGUF converter, which does not exist today.

### D2: build and prove both packs

Write the qwen35 GGUF path, then convert both quants. Accept: every tensor's dequantized values match the GGUF bit for
bit (the converter's manifest hash per tensor), the load prints no warnings, and the pack fits with the chosen KV.

### D3: first load

Serve it with one prompt. Accept: all layers on the card, the token embedding in page-locked RAM (2.4 GiB), coherent
output, no CUDA error.

### D4: the BF16 gate

`halo-bench --quality 100 --kld-base qwen38-27b-bf16-base.kld` (base PPL 6.7437, made from Unsloth's BF16 GGUF with
llama.cpp, sha256 `9ffb3bd9…`). Report mean, median, p99, top-1, PPL and the paired per-chunk interval for each quant and
each KV format that fits. Accept: the dense gate row exists, and the KV choice is made by whole-system KLD at depth,
then speed.

### D5: speed

Same fixtures and protocol as the sparse path, one repetition per arm, power limit recorded. Floors from the P0 run:
112 tok/s short decode, 118 at 36K, 93 at 137K; prompt 3,215 at 36K. Accept: the dense profile passes the floors with
the KV choice from D4.

### D6: identity and state

Greedy identity across the last two builds, and `qwen35_checkpoint_round_trip` with the real pack. Accept: 900 of 900
tokens and a state round trip that reproduces bytes.

### D7: beat NInfer and llama.cpp on quality

The comparison is one model, one quant, one card, one reference:

| Arm | Weights | What is measured |
|---|---|---|
| Halocline dense | Unsloth UD-IQ4_XS or UD-Q4_K_XL, repacked losslessly | KLD gate, speed, identity |
| llama.cpp | the same Unsloth GGUF, `llama-perplexity` and `llama-server` with the same KV | KLD gate, speed |
| NInfer | its own 27B format (it requantizes; the bytes differ) | KLD if it exposes the same gate, speed, sampled outputs |

Rules for the comparison:

- The KLD base is the same file for every arm. Lower mean KLD and higher top-1 is the win.
- Weights differ between arms because NInfer requantizes. Report that difference instead of hiding it; the claim is
  about the served result, not about identical bytes.
- Speed runs alternate arms in one session (ABBA), power limit fixed, one engine on the card.
- Quality beyond KLD uses the task suite in P2 on the same prompts, because KLD near the baseline does not rank models.
- Accept: a table with all three arms and intervals, and a statement of which arm wins on KLD, on the task set, and on
  speed. If Halocline does not win, that is the result and the README says so.

## P0: close the cheap gaps in the published build

These run beside P1 because they need GPU time, not converter work.

1. **Strata KLD.** Score Strata 0.2.10 on the same frozen base with its own pack and KV. Without it the speed table
   compares two engines and the quality comparison is missing. Accept: the 100-chunk row for Strata.
2. **Sampling distribution.** Seeded runs, top-1 and top-10 frequency within the run-to-run interval of the reference
   build. Accept: the same prompts, the same seeds, the numbers recorded.
3. **Long-context KLD.** Score windows placed at 64K, 137K and 200K of context, per KV format. This is the gate that
   would catch int8 KV drifting at depth. Accept: the table.
4. **Tool-call breadth.** Twenty or more prompts with a whether-to-call label. Accept: accuracy against the reference
   path.

## P2: task-level eval (SWE-bench)

Distributional distance near the baseline does not predict task outcomes, so the repo needs a task gate. SWE-bench is
the right first one: long context, tool use and code editing in one loop.

- **Harness:** mini-swe-agent, one scaffold for every arm, OpenAI-compatible endpoint against `bin/halo serve`.
- **Set:** SWE-bench Lite first (300 instances), then Verified (500). Pin the dataset revision and the images.
- **Settings:** temperature 0, one attempt per instance, 262,144 context, int8 KV, drafter on. Same flags for every arm.
- **Arms:** the sparse pack, the dense pack, then llama.cpp and NInfer on the same set if their harness support allows.
- **Report:** resolve rate with a 95% interval, tokens in and out, wall time per instance, and the KLD row for the
  same pack.
- **Blockers:** about 100 GB of image pulls, Docker with the NVIDIA runtime, hours per arm. Start with 25 instances.

Accept: a Lite run with a recorded resolve rate and no engine failures, repeated for each arm.

## P3: Qwen4 targets

Qwen4-Flash and Qwen4-27B are the stated targets. As of the research pass, Qwen4-Max is API-only and no Qwen4 weights
are public; Qwen4-27B's shape is a community lead. Do not plan against either until weights are on Hugging Face with a
config and a licence, the architecture matches a registered adapter or the diff is scoped, and a BF16 reference exists.
Then the order is P1 again: pack, load, KLD, speed, identity, task gate.

## P4: support and polish

- Remove the Windows and Docker code paths, or test them. Neither is supported today.
- Rewrite the carried-over user docs (`docs/configuration/`, `docs/models/`, `docs/precision/`); each still carries
  Basalt's banner.
- A release workflow: tag, archive, checksum, recorded toolchain.
- The formal trademark check for the name was never run; run it before commercial use.

## Order of work

D1 is decided, so D2 starts now. P0 runs in parallel because it needs the card, not a converter. D7 needs D3 to D5
first; P2 starts once the dense pack loads, so the task gate measures a model that runs. P3 waits on Qwen.
