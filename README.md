<p align="center">
  <img src="assets/halocline-hero.svg" alt="Halocline: the engine that hides nothing. A water column split by one bright boundary, labelled with the measured KLD to BF16, and a depth gauge with the decode speed measured at 36K and 137K tokens." width="100%">
</p>

<p align="center">
  <b>Halocline: the engine that hides nothing.</b><br>
  <b>One engine, two paths: sparse and dense.</b> Qwen mixture-of-experts and dense models on one NVIDIA Ada GPU
  (sm_89), a 24 GB RTX 4090. Each result names the run directory that produced it.
</p>

<p align="center">
  <code>status: pre-alpha</code> &nbsp; <code>target: sm_89 only</code> &nbsp; <code>models: Qwen4 family</code> &nbsp; <code>licence: MIT</code>
</p>

## Branches

This branch, `main`, is the default install and the one the rest of this README describes. Three optional research
branches add other weights for people who want them: install none, one or more, alone or beside `main`. Only `main` is
supported; the others are published for people to build and test on their own.
Each keeps its own docs, configs and data under `branches/<name>/`; an engine change a branch needs for its model is
listed in that folder's README.

| Branch | What it runs | Weights | Status |
|---|---|---|---|
| [`main`](https://github.com/T-Crypt/halocline/tree/main) | Both paths, sparse and dense, gated on KLD against BF16 | Unsloth UD-IQ4_XS (sparse); UD-Q4_K_XL and UD-IQ4_XS (dense), tensors copied byte for byte | supported, the default |
| [`halo-ista`](https://github.com/T-Crypt/halocline/tree/halo-ista) | The sparse path on ISTA-DASLab's per-tensor bit allocation, a smaller expert pool for Ada cards with less VRAM | ISTA-DASLab GSQ-RCO IQ3_S, and SC117's abliterated build of it | experimental, untested |
| [`halo-cyber`](https://github.com/T-Crypt/halocline/tree/halo-cyber) | The dense path on OrcaRouter's OrcaSAQ-2 Cyber 27B Uncensored, byte-identical, at its native 262,144 tokens | bring your own copy; the repository hosts none | research |
| [`halo-inkling`](https://github.com/T-Crypt/halocline/tree/halo-inkling) | Inkling 2 Mini (Thinking Machines' looped MoE, 26B with 3.4B active) on its own forward: greedy identical to llama.cpp, KLD 0.0032 against the released precision | Unsloth UD-Q4_K_XL, tensors copied byte for byte | experimental |

Each branch builds from its own clone and serves on its own ports (`main` 8092 and 8093, `halo-ista` 8094 and 8095,
`halo-cyber` 8096, `halo-inkling` 8097), so they never share a build, a pack or a port.

<img src="assets/halocline-divider.svg" width="100%" alt="">

## Two paths

| | **Sparse** | **Dense** |
|---|---|---|
| Model | Qwen3.8-Flash-Next: 125B, 6B active, 512 experts | Qwen3.8-27B: 64 layers, no experts (stands in for Qwen4-27B) |
| How it fits 24 GB | experts that miss the card live in page-locked RAM | every layer on the card; the token embedding in page-locked RAM |
| Weights | Unsloth UD-IQ4_XS v3.1, repacked | Unsloth UD-Q4_K_XL or UD-IQ4_XS v3.1, every tensor copied as stored |
| Mean KLD to BF16, top-1 | 0.08195, 90.45% | 0.00582, 96.67% (UD-Q4_K_XL); 0.01618, 94.31% (UD-IQ4_XS) |
| Against llama.cpp, same GGUF | | at parity: 0.00573 (f16 KV) and 0.01638 (q8_0 KV) |
| Prefill | 5,322 t/s at 16K, 5,070 at 64K | 3,383 t/s at 36K (UD-IQ4_XS); served, UD-Q4_K_XL: 3,309-3,325 t/s on a fresh 12-14K prompt, a median 2,220 on agent turns of 1,000+ new tokens over a warm cache |
| Decode, MTP drafter | 212 t/s over three fixtures | 113 t/s conversation, 98 prose (UD-IQ4_XS); served, UD-Q4_K_XL, a coding agent's replies of 200+ tokens at up to 73K context: median 117 t/s, p95 156, up to 166 ([`halocline-dense-served`](evals/2026-11-21/halocline-dense-served/)) |
| Context | 262,144, int8 KV | **262,144 served with q4_0 KV** by default (`--long-context`: K and V rotated 4-bit, +0.0011 mean KLD over K8V4 at 65K, 140K and 174K; a needle at 252,483 tokens found, [`halocline-long-context`](evals/2026-11-18/halocline-long-context/)); 175,616 with K8V4 KV as the lower-KLD lever ([two profiles](docs/models/qwen38-27b.md#two-profiles)) |
| Vision | `--vision dynamic`: the encoder on the card, 1.2 s median to the first token on an image ([`halocline-vision-dynamic`](evals/2026-11-20/halocline-vision-dynamic/)) | `--vision dynamic`: the encoder borrows the prompt workspace, 0.53 s median to the first token on an image against 14.4 s on the CPU ([`halocline-vision-lend`](evals/2026-11-21/halocline-vision-lend/)) |
| Status | **runs**, served through llama-swap | **runs**, gated, served at 262,144 tokens; about 10 times closer to BF16 than NInfer, 1-5% behind it on prose decode and up to 7% ahead on JSON at depth ([below](#dense-against-ninfer)) |
| Evidence | [`halocline-kv-kld`](evals/2026-11-17/halocline-kv-kld/), [community row](docs/community.md) | [`halocline-dense-quality`](evals/2026-11-17/halocline-dense-quality/), [`halocline-dense-load`](evals/2026-11-17/halocline-dense-load/), [`halocline-d8-kv`](evals/2026-11-17/halocline-d8-kv/), [`halocline-dense-served`](evals/2026-11-21/halocline-dense-served/) |

## Dense against NInfer

NInfer serves the same Qwen3.8-27B on the same card from its own requantized weights. I measure the two engines with
the same prompts, through the same llama-swap, one engine on the card at a time.

**Quality.** NInfer exposes no logits, so the comparison scores the same tokens on both and takes the gap to BF16 on
each target token (dNLL; lower is closer). At 65,536 tokens of context Halocline sits at +0.0018 and NInfer at +0.0199;
at 140,288, +0.0020 against +0.0201. Most of NInfer's distance comes from its weight conversion: with an 8-bit KV cache
it still reads +0.0172. Halocline copies Unsloth's tensors as stored.
[`ninfer-depth-ppl`](evals/2026-11-18/ninfer-depth-ppl/)

**Speed.** Thinking off on both, so neither engine splits its reply into a thinking pass and an answer; greedy, 1,200
tokens, streamed, decode rate as the client sees it, both at 262,144 tokens of context. Halocline is release aa156951
with the frequency-ranked draft head (`--draft-vocab`, 32,768 rows; FR-Spec, arXiv 2502.14856). The verify pass still
scores all 248,320 rows, so the output is the model's own. Both columns are from 2026-11-19:

| workload | Halocline t/s | NInfer t/s | Halocline vs NInfer |
|---|---|---|---|
| short prose | 91.7 | 92.7 | -1.1% |
| short JSON | 146.5 | 154.9 | -5.4% |
| 36K prose | 81.5 | 85.5 | -4.6% |
| 36K JSON | 143.5 | 136.9 | +4.8% |
| 137K prose | 68.7 | 70.7 | -2.8% |
| 137K JSON | 124.3 | 116.2 | +7.0% |

[`halocline-rc-aa156951`](evals/2026-11-19/halocline-rc-aa156951/),
[`halocline-ninfer-bar`](evals/2026-11-19/halocline-ninfer-bar/). The draft head moved prose from about -12% to within 5%
and cost JSON 3-5%. The remaining gap is round cost. On short prose Halocline keeps 2.30 tokens a round to NInfer's
2.21, but its round takes 25.1 ms to NInfer's 23.8. A fixed-width profile puts the extra time in the GEMVs: the
17,408-input down projection slows by 1.2 ms at four rows, and the small projections run at 71-76% of the card's
bandwidth ([`halocline-dense-profile-fixed`](evals/2026-11-19/halocline-dense-profile-fixed/)). Prefill trails as well,
16% slower at 36K and 31% at 137K.

Both paths share the build, the pack format, the server and the gates: the same BF16 KLD method, greedy identity, the
GPU suite. The sparse path trades quality for a model 4.6 times larger; the dense path keeps every weight on the card
and lands within llama.cpp's error bar on the same quant.

## Agent benchmark

[airbench](https://airbench.ai/) Benchmark v1.0 gives an agent 49 challenges (math, vision, email, purchasing, coding)
on one 2-hour clock. I ran both models through the HADAL harness on this card, served through llama-swap:

| | Sparse | Dense |
|---|---|---|
| Score | **49/49** | **48/49** |
| Time | 25m 00s | 1h 29m |
| Board, 2026-11-20 | #13 | #65 |
| Report | [64baefc1](https://airbench.ai/checkup/64baefc1-476e-41e8-ab11-938f248f8389/report) | [02db2264](https://airbench.ai/checkup/02db2264-4211-4dc1-93ba-da42b616e7cc/report) |

The dense model missed one vision challenge because the harness never told it that it could see. It worked all 19
images out with numpy and OCR and misread one arrow. I added the model's name and its image input to the harness
prompt. The sparse run spent about 340 s of its 15m41s vision section waiting on the CPU image encoder, so I moved
sparse to `--vision dynamic`: on the same 19 images, time to first token fell from 317 s to 22 s with the same replies
([`halocline-vision-dynamic`](evals/2026-11-20/halocline-vision-dynamic/)).

> [!IMPORTANT]
> **Halocline is for sm_89 (Ada) only, and it is tuned for one machine: [mine](SPECS.md).** The build refuses any compute
> capability except 8.9, and that will not change. It runs the Qwen4 family only. Its speeds and settings come from that
> one machine, so to reproduce them you need the same parts. On another Ada card it builds, and every number has to be
> measured again.
>
> Do not run the shipped flags on another host unchanged. The expert tier, the CPU share, the page-locked RAM and the
> n-gram table's disk reads all size to the parts. [The recipe](RESEARCH.md#using-halocline-as-a-base-for-your-own-card)
> lists the knobs and the order to re-measure in.

A halocline is the layer in deep water where salinity changes so sharply that the water above and below barely mix.
This repository works on one such boundary: the distance between what Qwen3.8-Flash-Next says at BF16 and what it says
from 4-bit weights, an int8 KV cache and a draft layer, on a single 24 GB card.

## What it is

|  |  |
|---|---|
| **Card** | One NVIDIA Ada GPU, compute capability 8.9: an RTX 4090, 24 GB, at 350 W. No other architecture is compiled, now or later. |
| **Host** | The machine in [SPECS.md](SPECS.md): i9-14900K (AVX2, AVX-VNNI), 64 GB RAM, Arch Linux, CUDA 13.4. |
| **Models** | The Qwen4 family only. Another architecture belongs in its own fork (`halocline-<family>`). |
| **Paths** | Sparse (Qwen3.8-Flash-Next: 512 experts, 10 routed plus 1 shared, expert offload) and dense (Qwen3.8-27B: 64 layers, no experts, no router). Both run. |
| **Sparse** | Qwen3.8-Flash-Next (`qwen4_exp`): 512 experts, 10 active plus 1 shared, QSA sparse attention, gated residual, n-gram embedding, MTP drafter. Experts that do not fit the card live in page-locked RAM. **Runs.** |
| **Dense** | Qwen3.8-27B (`qwen35`), standing in for Qwen4-27B until its weights exist: Gated DeltaNet and gated attention (3 to 1), SwiGLU, MTP drafter. Unsloth's v3.1 UD GGUFs repacked with every tensor as stored; the matrices run on llama.cpp's MMQ and MMVQ kernels in their stored types. **Runs, at 262,144 tokens by default** ([two profiles](docs/models/qwen38-27b.md#two-profiles)). |
| **Weights** | Unsloth GGUFs, converted to a pack that copies the quantized tensors byte for byte. One interim exception: the 195 `hc_*` tensors (Q8_0 in the GGUF) are cast to BF16. |
| **API** | OpenAI chat completions and Responses, Anthropic Messages, streaming and tool calls, and a stats page, all from one standalone server (`halocline serve`). One request at a time is the supported mode. Run it alone, or add it to [llama-swap](https://github.com/mostlygeek/llama-swap) as one entry ([Ada user?](#ada-user) step 6). No chat UI. |


## The question

Qwen3.8-Flash-Next holds [125B parameters with 6B active](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), plus a 51B
n-gram embedding table and a 4B MTP draft layer. At BF16 it does not fit one 24 GB card. Halocline serves it from
[Unsloth's UD-IQ4_XS GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): 93,672,359,738 bytes of weights,
18.2 GiB of experts in VRAM and 38.0 to 41.3 GiB page-locked in RAM, 262,144 tokens of context.

Each step away from BF16 carries a measurement:

| Step | Who sets it | What the repo measures |
|---|---|---|
| Expert and dense weights, UD-IQ4_XS | Unsloth's quant, copied byte for byte | mean KLD, top-1 and PPL against BF16 (`evals/2026-11-17/halocline-kv-kld/`) |
| KV cache format | Halocline, `--kv` | paired per-chunk dKLD with a 95% interval, prefill at 32K and 131K (same directory) |
| Draft layer | Halocline, `--spec` | decode speed by prompt type and tokens accepted per pass (`evals/2026-11-17/halocline-kv-kld/`) |
| Expert placement | Halocline, from the VRAM budget | greedy identity across builds (`evals/2026-11-17/halocline-greedy-identity/`) |

## The reference

One frozen BF16 file anchors every quality number:

| Item | Value |
|---|---|
| Model | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540), revision `de4b8e4d`: 48 layers, 512 experts (10 routed plus 1 shared), Gated DeltaNet and Qwen Sparse Attention, 262,144 native context |
| BF16 weights | [Unsloth BF16 GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/BF16), `Qwen3.8-Flash-Next-BF16-0000*-of-00008.gguf`; the base file's sha256 pins the score, not the repo revision |
| Base file | `flashnext-bf16-base.kld`, 12,664,728,820 bytes, sha256 `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9` |
| Made by | `llama-perplexity --kl-divergence-base`, llama.cpp `436f6f89e`, wikitext-2 `wiki.test.raw`, 100 chunks of 512 tokens, `-b 8192 -ub 8192`, base PPL 3.2749 (3.2733 on the scored rows) |
| Hosted at | [jesdga/basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases/tree/a65bae55f703e9c6dab6e9f335146c532fb7dd95), checksums in `SHA256SUMS` |
| Reported | mean, median and p99 KLD, top-1 agreement, PPL, and a paired per-chunk difference with a 95% t interval |

The base stores the tokens and the full-vocabulary log-probabilities of every scored position. It is never regenerated;
each numerics change is scored against the same bytes.

## Results

Qwen3.8-Flash-Next UD-IQ4_XS v3.1 with the MTP drafter, `--kv int8 --spec 8`, on the [reference machine](SPECS.md).
The quality and draft-window tables are [`evals/2026-11-17/halocline-kv-kld/`](evals/2026-11-17/halocline-kv-kld/); the speed
table is [`halocline-llama-swap`](evals/2026-11-17/halocline-llama-swap/) against the Strata column of
[`halocline-p0-floors`](evals/2026-11-16/halocline-p0-floors/); identity is
[`halocline-greedy-identity`](evals/2026-11-17/halocline-greedy-identity/).

### Quality against BF16

| KV | PPL | mean KLD | top-1 | paired dKLD vs int8 [95% CI] | prefill 131K |
|---|---|---|---|---|---|
| **int8** | 3.2323 | **0.08195** | **90.45%** | reference | **6,019 t/s** |
| fp16 | 3.2320 | 0.08230 | 90.33% | +0.00035 [-0.00061, +0.00131] | 2,997 t/s |
| k8v4 | 3.2424 | 0.08621 | 90.26% | +0.00426 [+0.00307, +0.00545] | 5,694 t/s |
| q4_0 | 3.2399 | 0.08843 | 90.03% | +0.00647 [+0.00499, +0.00795] | 5,700 t/s |

`--kv int8` is the setting. fp16's interval holds zero, so this gate cannot rank it ahead of int8, and fp16 halves the
131K prefill. k8v4 and q4_0 sit measurably further from BF16. The four formats span 0.0065 KLD; the weight quant's own
contribution is not isolated in this repo.

### Speed

| Test | Halocline | Strata 0.2.10, same card and quant | Ratio |
|---|---|---|---|
| Warm decode, short prompt | 148 to 182 tok/s | 95 to 101 tok/s | 1.6 to 1.8x |
| 36K needle: decode / prompt | 123 / 6,298 tok/s | 68 / 2,403 tok/s | 1.8x / 2.6x |
| 137K needle: decode / prompt | 121 / 5,664 tok/s | 87 / 2,864 tok/s | 1.4x / 2.0x |

Halocline ran MTP with int8 KV; Strata ran DFlash2 with q4_0 KV. The Strata column is its quiet-host rerun, and its
earlier runs plus the memory-pressure explanation are in the floors README. Nobody has run the KLD gate on Strata here,
so the table compares speed only.

### Identity across the scope removals

900 of 900 greedy tokens from `901fd9c9`, before the multi-GPU removals, to the published build. Both placed 7,961
experts in VRAM (18.23 GiB) and 41.30 GiB page-locked in RAM, with slot 0 verified.

### Draft window

| `--spec` | chat | structured | prose | structured tokens per pass |
|---|---|---|---|---|
| 5 | 134.8 | 238.1 | 136.3 | 5.04 |
| **8** | **137.5** | **266.0** | 133.0 | 7.80 |
| 12 | 133.4 | 267.9 | 133.3 | 7.80 |

Window 5 caps structured output 11% below window 8. Window 12 accepts no further than 8, so it buys nothing and costs
one extra draft step.

### Limits

- One machine. Nothing has been measured on another GPU, CPU or RAM size.
- The gate scores 512-token chunks, so it cannot see KV rounding that builds up deep in a long context. The needle
  checks at 36K and 137K cover retrieval there.
- The weight quant and the KV format are not separated: 0.0820 is the KLD of the whole 4-bit build.
- 195 `hc_*` tensors (Q8_0 in the GGUF) are cast to BF16 at conversion. That is an interim step, and the KLD numbers
  include it.
- The dense path loads its Q8_0 `ssm_alpha`/`ssm_beta` as BF16 (the gating kernels read BF16): the one rounding on that
  path, at load; the pack holds the GGUF's bytes. Its 262K context is open (roadmap D8).

## How a change gets in

Every engine commit in this history passed these gates, and its logs and hash files are in the repo:

1. A clean build for sm_89 (728 targets).
2. SASS hashes per object and per function (`gates/`): kernels whose source did not change come out byte-identical, and
   inside a changed object only the functions the change targets may differ.
3. The CPU suite (86 tests) and the GPU suite (122 tests, 7 skip on sm_89), plus the Python server suite (1,079 tests).
4. For numerics changes: paired KLD against the frozen BF16 base, and greedy identity against the previous build.

The engine ships every test and the data behind every number: an `evals/` tree with the command, the raw JSON and the
log that made each measurement, `gates/` with the SASS hashes per step, and `logs/` with the ctest and build logs.

## Ada user?

You have an RTX 40-series card or an Ada workstation or server card and want to try this. Halocline is pre-alpha, with
one maintainer and one reference machine, and its numbers hold on that machine until you measure your own. The steps
below take you from a clean checkout to a validated pack and a posted bench.

**1. Check the card.** `nvidia-smi --query-gpu=name,compute_cap --format=csv` has to print `8.9`. The build refuses
anything else, and that stays. You need Linux x86-64 and the CUDA toolkit (the reference machine runs 13.4). The dense
packs need a 24 GB card: RTX 4090, RTX 5000 or 6000 Ada, L40, L40S. Nobody has measured a smaller Ada card yet.

**2. Build**, with the tests:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON
ninja -C build
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**3. Test**, with nothing else on the card:

```sh
ctest --test-dir build -L cpu -j 16                          # host logic, formats, tokenizers
ctest --test-dir build -L gpu -j 1                           # kernels against CPU references; logs/ctest-gpu.sm89.log lists the sm_89 skips
.venv/bin/python -m unittest discover -s serve/tests -t .    # the server
```

**4. Convert a pack.** Start with the dense 27B: it lives on the card, so the host RAM and the expert tiers do not enter
into it. Download `Qwen3.8-27B-UD-Q4_K_XL.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), then:

```sh
.venv/bin/python tools/convert/convert.py --gguf Qwen3.8-27B-UD-Q4_K_XL.gguf --no-vision --out Qwen3.8-27B-UD-Q4_K_XL.halo
.venv/bin/python tools/convert/gguf_qwen35.py verify Qwen3.8-27B-UD-Q4_K_XL.halo Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The sparse model needs 64 GiB of RAM and about 190 GiB of disk; its commands are under [Build and run](#build-and-run).

**5. Validate.** Fetch `qwen38-27b-bf16-base.kld` (12.7 GB) from
[basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases), check its SHA-256 against
[the precision page](docs/precision/qwen38-27b.md), and score your pack:

```sh
bin/halocline-bench --model Qwen3.8-27B-UD-Q4_K_XL.halo --quality 100 --kld-base qwen38-27b-bf16-base.kld -- --kv int8
```

The reference machine reads mean KLD 0.00582 and top-1 96.67%. The pack copies the GGUF's tensors as stored, so your
result should land close to that. A clear gap points at the conversion or a bug: open an issue with the full output.

**6. Serve.** `halocline serve` is a standalone server and needs nothing else:

```sh
bin/halocline serve Qwen3.8-27B-UD-Q4_K_XL-vision.halo --host 127.0.0.1 --port 8093 --spec 8 --long-context --vision dynamic
```

That is the default profile: 262,144 tokens on rotated 4-bit K and V. For the lower-KLD profile at 175,616 tokens,
replace `--long-context` with `--max-context 175616 --kv k8v4 --mtp-kv int8`; `examples/` has both as config files.

OpenAI and Anthropic clients talk to `http://127.0.0.1:8093/v1`, and the same address in a browser shows the stats page.
It binds to loopback by default. To expose it, read [security](docs/configuration/index.md) and set `--api-key`. To run
it beside other models under [llama-swap](https://github.com/mostlygeek/llama-swap), add one entry whose `cmd` is that
line and whose `proxy` is that port.

**7. Post a bench.** `bin/halocline-bench --community --model PACK.halo --submitter YOUR_NAME` runs the fixed preset
and adds one row to `docs/community/benchmarks.json`. The row records the card, its power limit, the PCIe link, the CPU
and the RAM. Open a pull request with that file, and paste the step 5 summary lines in the description if you ran it.
[docs/community.md](docs/community.md) explains the preset. The preset has only run on the sparse pack so far, so the
first 27B row also tests it on the dense path: say so in the pull request if it fails.

**8. Contribute.** Read [CONTRIBUTING.md](CONTRIBUTING.md) before you write code. It lists the work this repo takes and
the work it closes. In short: sm_89 only, the Qwen4 family only, and no change merges that slows the reference profiles
or moves their KLD past the paired interval.

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build                       # binaries land in bin/: halocline, halocline-bench, halocline-probe, halocline-vision
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# the draft layer: Qwen's BF16 mtp.* tensors, hash-checked
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn

# the pack: Unsloth's GGUF shards, quantized tensors copied as published
.venv/bin/python tools/convert/convert.py --gguf Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
    --no-vision --compat-bf16 --mtp-source mtp-qfn --out QFN-UD-IQ4_XS.halo

# serve, with the settings the evals above used
bin/halocline serve QFN-UD-IQ4_XS.halo --host 127.0.0.1 --port 8092 \
    --spec 8 --prefill auto --max-context 262144 --kv int8

# the quality gate against the BF16 base
bin/halocline-bench --model QFN-UD-IQ4_XS.halo --quality 100 --kld-base flashnext-bf16-base.kld -- --kv int8 --prefill auto
```

The pack format is version 2, the layout Halocline inherited (`.halo`, magic `HALOCLMF`). Every pack written before the
rename carries `BASALTMF` and still loads; nothing rewrites a pack in place, so a `.basalt` file keeps its name and
the engine reads it as it is.

On any other Ada card, `bin/halocline-bench --community --model PACK.halo --submitter NAME` runs the fixed preset, checks the
card and prints one row for [the community table](docs/community.md). That page lists the supported cards, what each VRAM
size does to the expert pool, and what a row records.

## Repository

| Path | What |
|---|---|
| `src/`, `include/halo/`, `serve/`, `tools/` | the engine, the Python server, the converter and the bench |
| `evals/` | every measurement: command, raw JSON, logs, and the script that made it |
| `gates/` | SASS hashes per step, whole objects and per function |
| `logs/` | build and test logs per step |
| `docs/design/` | the build spec and the review notes the engine was planned from |
| [`docs/roadmap.md`](docs/roadmap.md) | the ordered open work, with accept criteria |
| [`docs/testing.md`](docs/testing.md) | every gate, what it proves, and what is still missing |
| `docs/research/` | the research notes behind the design, with sources |
| [`SPECS.md`](SPECS.md) | the reference machine |
| [`RESEARCH.md`](RESEARCH.md) | the recipe and the sources, applied or read |
| [`LIFTS.md`](LIFTS.md) | the base snapshot's hashes and what came from where |
| `docs/` | user docs; the pages still carried over from Basalt are the next to rewrite (open work 6) |

## Sources

| Source | What this repo took from it |
|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at `de4b8e4d`, [technical report](https://arxiv.org/abs/2608.30320) | the model, its 31 BF16 `mtp.*` tensors, and the architecture the adapter implements |
| [Unsloth GGUFs](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/814db05f88b3b827ea7cfcab5344fd218de494d2) at `814db05f`, [BF16 shards](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/BF16) | the UD-IQ4_XS weights and the BF16 weights the reference was scored from |
| [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) at `a65bae55` | the frozen BF16 KLD base |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `436f6f89e`, [perplexity tool](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) | `--kl-divergence-base`, the quantization formats and the reference quantizers |
| [arXiv 2606.19558](https://arxiv.org/abs/2606.19558), Displacement Is Not Direction | KLD tracks benchmark scores across a quant cohort and loses that link near the baseline, so each serve also runs needle retrieval at 36K and 137K |
| [arXiv 2608.06564](https://arxiv.org/abs/2608.06564), Which Decisions Low-Bit Quantization Breaks | low-bit quantization flips whether to call a tool more often than which tool, so the smoke test keeps a tool-call check |

[`RESEARCH.md`](RESEARCH.md) lists the papers read but not applied, with the reason for each.

## Open work

The ordered list, with status, accept criteria and blockers, is in [`docs/roadmap.md`](docs/roadmap.md). The testing that
is missing is in [`docs/testing.md`](docs/testing.md). The short list:

1. **Dense round cost (Qwen3.8-27B).** At 262K the dense path trails NInfer by 1-5% on prose and leads JSON at depth
   ([above](#dense-against-ninfer)). The time goes to the verify round: the down projection's multi-row GEMV at four
   and five rows first, then the small projections at 71-76% of bandwidth, then the 262K prefill (16% and 31% behind
   at 36K and 137K).
2. **Task-level eval.** SWE-bench Lite, one scaffold, the same flags for every arm.
3. **Gaps in the current gates.** Strata KLD, sampling distribution, KLD at 64K to 200K of context, a broader tool-call
   set.
4. **Remove the last sm_90+ code:** the decode cluster kernels (sampler, QSA top-k), the fused-verify chain (GDN cluster
   recurrence, mid chain, dense tensor-core path, gated-residual fusion), the in-engine model router, the Windows and
   Docker files.
5. A DFlash2 drafter A/B against MTP, and the `hc_*` BF16 cast revisited against the KLD numbers.
6. Rewrite the carried-over docs.

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

<p align="center"><sub>Measurements from the reference machine, 2026-11-16 and 2026-11-17. The repository holds the runs.</sub></p>
