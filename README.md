<p align="center">
  <img src="assets/halocline-hero.svg" alt="Halocline: the engine that hides nothing. A water column split by one bright boundary, labelled with the measured KLD to BF16, and a depth gauge with the decode speed measured at 36K and 137K tokens." width="100%">
</p>

<p align="center">
  <b>An inference engine for one machine and one model family, built to stay as close to BF16 as the weights allow.</b><br>
  The Qwen4 family on one NVIDIA Ada GPU (sm_89). Every number links to the run that produced it.
</p>

<p align="center">
  <code>status: pre-alpha</code> &nbsp; <code>target: sm_89 only</code> &nbsp; <code>models: Qwen4 family</code> &nbsp; <code>licence: MIT</code>
</p>

<img src="assets/halocline-divider.svg" width="100%" alt="">

> [!IMPORTANT]
> **Halocline is for sm_89 (Ada) only, and it is tuned for one machine: [mine](SPECS.md).** It will never support
> another GPU architecture; the build refuses anything but compute capability 8.9. It runs the Qwen4 family only. Its
> speeds and settings come from that one machine, so to reproduce them you need the same parts. On another Ada card it
> builds, and every number has to be measured again.
>
> To build an engine like this for your own card, read [the recipe](RESEARCH.md): the steps, the gates, the papers
> behind each decision, and how to fork Halocline and swap in kernels for your architecture.

A halocline is the layer in deep water where salinity changes so sharply that the water above and below barely mix.
This engine is built around one boundary of its own: the distance between what the model says at full precision and
what it says here. Each change is measured against that line, and the measurements live in this repo.

## What it is

|  |  |
|---|---|
| **Card** | One NVIDIA Ada GPU, compute capability 8.9: an RTX 4090, 24 GB, at 350 W. No other architecture is compiled, now or later. |
| **Host** | The machine in [SPECS.md](SPECS.md): i9-14900K (AVX2, AVX-VNNI), 64 GB RAM, Arch Linux, CUDA 13.4. |
| **Models** | The Qwen4 family only. Another architecture belongs in its own fork (`halocline-<family>`). |
| **Sparse** | Qwen3.8-Flash-Next (`qwen4_exp`): 512 experts, 10 active plus 1 shared, QSA sparse attention, gated residual, n-gram embedding, MTP drafter. Experts that do not fit the card live in page-locked RAM. **Runs.** |
| **Dense** | Qwen3.8-27B, standing in for Qwen4-27B until its weights exist. **Does not run yet.** |
| **Weights** | Unsloth GGUFs, converted to a pack that copies the quantized tensors byte for byte. One interim exception: the 195 `hc_*` tensors (Q8_0 in the GGUF) are cast to BF16. |
| **API** | OpenAI-compatible chat completions with streaming and tool calls, one request at a time, meant to sit behind [llama-swap](https://github.com/mostlygeek/llama-swap). |

## Measured so far

Qwen3.8-Flash-Next UD-IQ4_XS v3.1 with the MTP drafter, int8 KV, on the [reference machine](SPECS.md).

| Measure | Value | Evidence |
|---|---|---|
| Mean KLD to BF16 (100 chunks of 512) | 0.0820, top-1 90.45%, PPL 3.232 (BF16 3.273) | [`evals/2026-11-17/halocline-kv-kld`](evals/2026-11-17/halocline-kv-kld/) |
| KV format | int8: fp16 is no closer (paired +0.0004, interval holds 0) and halves 131K prefill; k8v4 and q4_0 measurably further | same |
| Decode, warm, short prompt | 148 to 182 tok/s | [`evals/2026-11-17/halocline-llama-swap`](evals/2026-11-17/halocline-llama-swap/) |
| 36K needle: decode / prompt | 123 / 6,298 tok/s | same |
| 137K needle: decode / prompt | 121 / 5,664 tok/s | same |
| Same card and quant, Strata 0.2.10 (DFlash2, q4_0 KV) | 95 to 101 warm; 68 / 2,403 at 36K; 87 / 2,864 at 137K | [`evals/2026-11-16/halocline-p0-floors`](evals/2026-11-16/halocline-p0-floors/) |
| Greedy identity across the scope removals | 900 of 900 tokens | [`evals/2026-11-17/halocline-greedy-identity`](evals/2026-11-17/halocline-greedy-identity/) |

Limits: the KLD gate scores 512-token chunks, so it cannot see KV rounding that builds up deep in a long context; the
needle checks cover retrieval there. Strata's column is its quiet-host rerun; its earlier runs and the reason are in
the floors README.

## How a change gets in

Each commit in this history passed these gates, and its logs and hash files are in the repo:

1. A clean build for sm_89.
2. SASS hashes per object and per function (`gates/`): kernels whose source did not change come out byte-identical,
   and inside a changed object only the functions the change targets may differ.
3. The CPU suite (86 tests) and the GPU suite (122 tests, 7 skip on sm_89), plus the Python server suite (1,079 tests).
4. For numerics changes: paired KLD against the frozen BF16 base, and greedy identity against the previous build.

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build                       # binaries land in bin/: halo, halo-bench, halo-probe, halo-vision
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# the draft layer: Qwen's BF16 mtp.* tensors, hash-checked
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn

# the pack: Unsloth's GGUF shards, quantized tensors copied as published
.venv/bin/python tools/convert/convert.py --gguf Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
    --no-vision --compat-bf16 --mtp-source mtp-qfn --out QFN-UD-IQ4_XS.basalt

# serve, with the settings the evals above used
bin/halo serve QFN-UD-IQ4_XS.basalt --host 127.0.0.1 --port 8092 \
    --spec 8 --prefill auto --max-context 262144 --kv int8

# the quality gate against a BF16 KLD base
bin/halo-bench --model QFN-UD-IQ4_XS.basalt --quality 100 --kld-base flashnext-bf16-base.kld -- --kv int8 --prefill auto
```

The pack format is Basalt's model-file format v2, unchanged (`.basalt`, magic `BASALTMF`), so converted packs keep
their names.

## Repository

| Path | What |
|---|---|
| `src/`, `include/halo/`, `serve/`, `tools/` | the engine, the Python server, the converter and the bench |
| `evals/` | every measurement: command, raw JSON, logs, and the script that made it |
| `gates/` | SASS hashes per step, whole objects and per function |
| `logs/` | build and test logs per step |
| `docs/design/` | the build spec and the review notes the engine was planned from |
| `docs/research/` | the research notes behind the design, with sources |
| [`SPECS.md`](SPECS.md) | the reference machine |
| [`RESEARCH.md`](RESEARCH.md) | the recipe and the sources, applied or read |
| [`LIFTS.md`](LIFTS.md) | the base snapshot's hashes and what came from where |
| `docs/` | user docs carried over from Basalt, each marked for a rewrite |

## Open work

- Remove the last sm_90+ code: the decode cluster kernels (sampler, QSA top-k), the fused-verify chain (GDN cluster
  recurrence, mid chain, dense tensor-core path, gated-residual fusion), the in-engine model router, the Windows and
  Docker files.
- The dense path (Qwen3.8-27B) on the GGUF weights.
- A DFlash2 drafter A/B against MTP on Halocline.
- Revisit the `hc_*` BF16 cast against the KLD numbers.
- Rewrite the carried-over docs.

## Design rules

Targets the engine is being built toward. Not all of them exist in code yet.

- One architecture, one toolchain, recorded per release.
- Every choice that affects speed or output (expert cache, prefill mode, drafter, KV format) resolved once per model
  into a checked-in profile, never keyed on a model's name.
- Expert swaps copy exact bytes, so where an expert lives changes speed, never text.
- A regression in mean or tail KLD beyond its interval, or in speed beyond run-to-run range, fails the release.

## Lineage

Halocline starts from a one-time snapshot and has no upstream after it. Credit where the code came from:

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT), the expert-offload engine the rest
  descends from.
- [Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo (MIT), the multi-model engine forked from Strata.
- [basalt_ada](https://github.com/Ex3c4Def/basalt_ada) (MIT), the Ada port of Basalt this snapshot merges.
- [Infernix](https://github.com/Wallawalla47/Infernix) (Apache-2.0), tile layouts in several kernels.
- [Unsloth](https://huggingface.co/unsloth) for the GGUFs, and [Qwen](https://huggingface.co/Qwen) for the models, which
  carry their own licences.

Original notices are kept in `LICENSE` and `THIRD_PARTY_LICENSES`; [`LIFTS.md`](LIFTS.md) records the snapshot.

<img src="assets/halocline-divider.svg" width="100%" alt="">

<p align="center"><sub>Built for one machine. Measured, not guessed.</sub></p>
