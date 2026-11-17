# Qwen3.8-Flash-Next (sparse)

The sparse path, and the only model that runs today. Qwen3.8-Flash-Next is a 125B-parameter mixture-of-experts model
with 6B activated per token, plus a 51B n-gram embedding table and a 4B MTP layer
([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[report](https://arxiv.org/abs/2608.30320)).

| Property | Value |
|---|---|
| Architecture id | `qwen4_exp` |
| Layers | 48: 36 Gated DeltaNet, 12 Qwen Sparse Attention |
| Experts | 512 per layer, 10 routed plus 1 shared, intermediate dimension 640 |
| Hidden size | 2,560 |
| Context | 262,144 native |
| Vocabulary | 248,320 |
| Drafting | MTP layer, `--spec 8` |

## Pack

From Unsloth's UD-IQ4_XS v3.1 shards at revision `814db05f`, 93,672,359,738 bytes:

```sh
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn
.venv/bin/python tools/convert/convert.py --gguf Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
    --no-vision --compat-bf16 --mtp-source mtp-qfn --out QFN-UD-IQ4_XS.halo
```

The pack copies the quantized tensors byte for byte. The draft layer adds 0.77 GiB (sections `mtp/dense` and
`mtp/experts`); the 31 BF16 source tensors are 4.9 GiB and are hash-checked per tensor. The 195 `hc_*` tensors are cast
from Q8_0 to BF16 at conversion (`--compat-bf16`), an interim step the KLD numbers include.

## Run

```sh
bin/halocline serve QFN-UD-IQ4_XS.halo --host 127.0.0.1 --port 8092 \
    --spec 8 --prefill auto --max-context 262144 --kv int8
```

Placement on the reference machine: 18.2 GiB of experts in VRAM (7,961 slots), 38.0 to 41.3 GiB page-locked in RAM, 0 on
SSD. About 4 GiB of system RAM stays free with the model loaded.

## What is measured

- KLD against BF16 and the KV choice: [README](../../README.md#quality-against-bf16) and
  [`evals/2026-11-17/halocline-kv-kld`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-kv-kld).
- Speed, including the needle checks at 36K and 137K:
  [`halocline-llama-swap`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-llama-swap).
- Greedy identity across the removals:
  [`halocline-greedy-identity-final`](https://github.com/T-Crypt/halocline/tree/main/evals/2026-11-17/halocline-greedy-identity-final).
- The roundings and the gate: [precision](../precision/qwen38-flash-next.md).

## Not in use

The reference pack is built `--no-vision`, so images and video are off. Multi-GPU, the peer expert tier, NVFP4 packs and
the sm_120 kernels were removed with the sm_89 scope; Basalt's changelog records them.
