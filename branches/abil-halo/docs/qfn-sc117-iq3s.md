# Qwen3.8-Flash-Next, SC117 GSQ-RCO abliterated IQ3_S (branch `abil-halo`)

This branch adds one more sparse source next to the Unsloth UD-IQ4_XS pack on `main`: SC117's abliterated build of
ISTA-DASLab's GSQ-RCO IQ3_S tier. `main` stays Unsloth-only and keeps the official gates. This page and its evals live
on `abil-halo` only.

> [!WARNING]
> The model is abliterated: its refusal direction is removed, so it has no reliable built-in guardrails. Serve it only
> behind your own access control, and never give it a router or role alias that other agents pick up on their own.

| Property | Value |
|---|---|
| Source | [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF), revision `90a3350c`, folder `IQ3_S/` |
| Quantization | [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (GSQ, arXiv 2604.18556; RCO, arXiv 2605.00649): learned per-tensor grids and scales, one type per tensor under a size budget |
| Abliteration | 144 tensors (`ssm_out`, `attn_output`, `ffn_down_shexp`, `ffn_down_exps`, all 48 layers) transplanted from [orcarouter's uncensored release](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF); the other 1,079 tensors match upstream byte for byte (SC117's per-tensor blake2b) |
| Shard 1, weights | `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf`, 55,063,448,064 bytes, sha256 `40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c` |
| Shard 2, n-gram table | `...-00002-of-00002.gguf`, 28,800,138,432 bytes, sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` (byte-identical to upstream GSQ-RCO) |
| Licence | Apache-2.0, from the base model |

## Tensor types

The IQ3_S tier mixes 13 types (SC117's histogram): BF16 484, F32 292, Q8_0 96, Q6_K 89, Q4_K 43, IQ4_XS 45, IQ2_S 40,
IQ4_NL 39, IQ3_XXS 34, Q5_K 31, IQ3_S 20, Q2_0 9, F16 1. The routed experts use IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, IQ4_NL
and Q2_0; `ffn_down_exps` rows hold 640 values, so only 32-value block types fit there (Q2_0 on 9 layers, IQ4_NL on 39).

`qwen4exp` already reads each of these on the card (`src/prefill/moe_mmq.cu`, `src/models/qwen4exp/prefill/moe_fused_iq.cu`,
`src/models/qwen4exp/kernels/dec_moe.cuh`). The CPU expert rows (`src/kernels/cpu/iq_avx2_rows.inc`) cover IQ3_S,
IQ3_XXS, IQ2_S and IQ4_XS; IQ4_NL and Q2_0 experts in the RAM tier take the ggml-cpu dot products. Only the IQ3_S tier is
supported here; SC117's IQ3_XXS, IQ2_XS and Q2_0 tiers are not.

## Pack

SC117's MTP head is the stock base model's, not abliterated, so the draft layer comes from Qwen's BF16 checkpoint
(`mtp_fetch.py`) as for ISTA's build, and the projector from ISTA's repo (the same Flash-Next BF16 `clip` model):

```sh
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn
.venv/bin/python tools/convert/convert.py --gguf sc117/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf \
    --mmproj ista/mmproj-Qwen3.8-Flash-Next-BF16.gguf --mtp-source mtp-qfn \
    --name QFN-SC117-GSQ-RCO-abliterated-IQ3_S --out QFN-SC117-GSQ-RCO-abliterated-IQ3_S-vision.halo
```

The reference pack was built with `--sections-from` the UD pack instead. For ISTA's build the two routes give the same
bytes in every section but `meta` and the projector, whose 334 tensors match and whose GGUF header names a different
publisher.

The pack is 87,889,249,268 bytes; every one of the tier's 1,223 tensors converts.

## Status

Experimental and untested, like the rest of this branch: no KLD against BF16 and no comparison with `main`. The
abliteration changes outputs on purpose.

## Serving

[`../examples/qfn-sc117-iq3s.json`](../examples/qfn-sc117-iq3s.json) is the local config: its own port and alias, the served sparse flags. Give it a
llama-swap entry of its own with no router or role aliases.
