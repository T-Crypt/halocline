# The reference machine

Halocline is built, tuned and measured on one machine. Every number in this repo comes from it. To get the same
numbers you need the same parts; a different card, CPU, RAM size or disk gives you different speeds, and on anything
other than sm_89 the engine does not build.

Recorded 2026-11-17. A change to any line here is recorded with a date, and the evals it affects say which spec they ran
on.

## Hardware

| Part | Value | Why it matters to Halocline |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9 | the only target the build accepts; about 18 GiB of experts fit beside the dense weights, KV and drafter |
| GPU power limit | 350 W (set by the operator; the card allows more) | decode and prefill speed |
| GPU link | PCIe 4.0 x16 | experts that live in RAM stream over it during prefill |
| GPU clocks (max) | SM 3,285 MHz, memory 10,501 MHz | speed |
| VBIOS | 95.02.3C.40.B9 | |
| CPU | Intel Core i9-14900K, 32 threads, AVX2 and AVX-VNNI, no AVX-512 | the CPU expert kernels run on AVX2; prefill gives the CPU 0.6 of the streamed experts |
| RAM | 64 GB (62.5 GiB visible) | the Flash-Next pack page-locks 41 GiB of experts; about 4 GiB stays free with the model loaded |
| Pack disk | NVMe: TEAM TM8FP6001T 1 TB, PCIe 3.0 x4, ext4 (since 2026-11-19) | the sparse path reads the n-gram table from the pack during prefill, 4 KiB at a time; with a cold page cache a 137K prompt reads 4,106 to 4,417 t/s here and 1,857 to 3,262 from the WD SATA SSD the packs lived on before ([`halocline-sparse-abba`](evals/2026-11-19/halocline-sparse-abba/)) |
| Other disks | WD SATA SSD, 2 TB (`WDS200T2B0B`), ext4, also the system disk; KIOXIA KXG60ZNV512G 512 GB NVMe (Windows) | the dense packs (the dense path reads its pack once, at load); the GGUF shards conversion reads sit on the TEAM NVMe |

## Software

| Part | Value |
|---|---|
| OS | Arch Linux, kernel 7.2.14 |
| NVIDIA driver | 615.78.08 |
| CUDA toolkit | 13.4 (V13.4.92) |
| Host compiler | GCC 16.2.1 |
| CMake | 4.4.5 |
| Build | `-DCMAKE_CUDA_ARCHITECTURES=89`, Release |
| Memory plan | engines get a 55 GiB floor, the desktop is capped, earlyoom stops the largest process below about 2.5 GiB free |
| Front door | llama-swap, one model loaded at a time, under a Berth GPU lease |

## Models

| Model | Weights | Role |
|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`qwen4_exp`) | [Unsloth UD-IQ4_XS v3.1](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/814db05f88b3b827ea7cfcab5344fd218de494d2), four shards, 93,672,359,738 bytes | the sparse path |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) at `4af51b35` (UD-IQ4_XS and UD-Q4_K_XL on disk) | stand-in for Qwen4-27B until its weights exist; the dense path runs and serves 262,144 tokens by default |
| MTP draft layer | the 31 BF16 `mtp.*` tensors of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) at `de4b8e4d`, 4.9 GiB | drafter for the sparse path |
| BF16 KLD base | [`flashnext-bf16-base.kld`](https://huggingface.co/datasets/jesdga/basalt-kld-bases/tree/a65bae55f703e9c6dab6e9f335146c532fb7dd95), sha256 `c4e0221632607324d314556b8813d772867c300454ecdea37f1bf9d5555cf1a9`, from `llama-perplexity --kl-divergence-base`, wikitext-2, 100 chunks of 512, base PPL 3.2749 (3.2733 on the scored rows) | the quality reference |

Model facts for the sparse path, from the model's `config.json` at `de4b8e4d`: 48 layers, 12 cycles of three Gated
DeltaNet layers and one Qwen Sparse Attention layer, hidden size 2,560, 512 experts with 10 routed and 1 shared active
per token, expert intermediate dimension 640, n-gram vocabulary 20,000,000 at layer 2, 262,144 native context. Qwen
reports 125B parameters with 6B activated, plus 51B in the n-gram embedding and 4B in the MTP layer.

## What changes when your machine differs

- **Another GPU architecture:** the build refuses it. Halocline compiles sm_89 code only.
- **Another Ada card** (4080, 4070, L4, L40): it builds, but the expert split between VRAM and RAM, the expert cache
  size and every speed number change with VRAM, bandwidth and SM count. Nobody has measured KLD on another card;
  re-run the quality gate before you trust any number from this repo.
- **Less RAM:** the 41 GiB of page-locked experts plus the OS will not fit, and experts spill to disk. Not measured.
- **The pack disk:** the sparse path keeps reading the pack after load (the n-gram table, 16 rows a token, one 4 KiB read
  each), so a slow or shared disk lowers prefill at depth whenever the page cache does not hold those pages. Put the
  sparse pack on a fast NVMe that nothing else is busy on.
- **AVX-512:** the CPU expert kernels have an AVX-512 path this machine does not exercise; it is untested here.
