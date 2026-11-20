# Sparse sources (branch `halo-ista`)

This branch converts two builds of ISTA-DASLab's GSQ-RCO IQ3_S quant of Qwen3.8-Flash-Next. Each becomes its own pack
and runs under its own name. Unsloth's UD-IQ4_XS lives on [`main`](https://github.com/T-Crypt/halocline/tree/main),
which keeps its gates; it appears below only for the expert sizes.

| Source | Download (folder, revision) | Pack | Local config | llama-swap name |
|---|---|---|---|---|
| [ISTA-DASLab GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | `IQ3_S/` and `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, `ed59f920` | `QFN-GSQ-RCO-IQ3_S-vision.halo` | [`qfn-gsq-iq3s.json`](../examples/qfn-gsq-iq3s.json) (port 8094) | `Halocline-QFN-GSQ-IQ3_S` |
| [SC117 GSQ-RCO abliterated IQ3_S](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) | `IQ3_S/`, `90a3350c`, plus ISTA's projector | `QFN-SC117-GSQ-RCO-abliterated-IQ3_S-vision.halo` | [`qfn-sc117-iq3s.json`](../examples/qfn-sc117-iq3s.json) (port 8095) | `Halocline-QFN-SC117-IQ3_S` |

SC117's build has no reliable refusals. Read the warning on [its page](qfn-sc117-iq3s.md) before you serve it, and
give it no router or role alias.

| File | Bytes | sha256 |
|---|---|---|
| ISTA `IQ3_S/...-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/...-00002-of-00002.gguf`, the n-gram table, the same bytes in both repos | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| ISTA `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |

## Convert

Neither GGUF carries a drafter. `mtp_fetch.py` reads Qwen's 31 BF16 `mtp.*` tensors (4.9 GB) out of the BF16
checkpoint by range requests and checks each hash. The converter takes shard 1 and finds shard 2 beside it.

```sh
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn

# ISTA-DASLab GSQ-RCO IQ3_S
.venv/bin/python tools/convert/convert.py --gguf ista/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --mmproj ista/mmproj-Qwen3.8-Flash-Next-BF16.gguf --mtp-source mtp-qfn \
    --name QFN-GSQ-RCO-IQ3_S --out QFN-GSQ-RCO-IQ3_S-vision.halo

# SC117 GSQ-RCO abliterated IQ3_S
.venv/bin/python tools/convert/convert.py --gguf sc117/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf \
    --mmproj ista/mmproj-Qwen3.8-Flash-Next-BF16.gguf --mtp-source mtp-qfn \
    --name QFN-SC117-GSQ-RCO-abliterated-IQ3_S --out QFN-SC117-GSQ-RCO-abliterated-IQ3_S-vision.halo
```

## How they differ

Both IQ3_S packs run on the same engine, flags and sm_89 kernels as `main`'s UD pack. They differ from it in the
routed experts' types and size:

| Pack | gate and up (layers) | down (layers) | MB per routed expert |
|---|---|---|---|
| UD-IQ4_XS (`main`) | IQ3_XXS 28, IQ4_XS 17, IQ2_S 3 | Q5_0 33, IQ4_NL 13, Q2_0 2 | 2.457 |
| GSQ-RCO IQ3_S, both builds | IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1 | IQ4_NL 39, Q2_0 9 | 2.046 |

The two IQ3_S packs match layer for layer in expert types; SC117 changed the values. This branch is experimental and
untested: nobody scores these packs against BF16 or compares them with `main`. For quality, read ISTA-DASLab's model
card.
