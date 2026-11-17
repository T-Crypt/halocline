# Community runs

Every row in `docs/community/benchmarks.json` was produced by the same command on someone's own card:

```sh
halocline-bench --community --model PACK.halo --submitter YOUR_NAME
```

The preset is fixed: three decode fixtures (`conv`, `struct`, `prose`), a 16K and a 64K prompt, each measured cold in
its own engine process. The bench refuses a flag that would change it, because a row with different flags would not
compare with the others. `halocline-bench --help` prints the fixture contents and how to regenerate them.

## What the preset needs

One Ada card. Halocline builds sm_89 code only: RTX 4090, 4080, 4070 Ti, 4070, the RTX 4000/4500/5000/6000 Ada
workstation cards, and the L4, L40 and L40S servers. Ampere, Turing, Hopper and Blackwell cards have no code in the
binary, and the bench refuses them before it starts. A MIG slice is not one card and is not supported.

Linux x86-64 and a driver new enough for CUDA 12. There is nothing else to install. The bench reads the card's compute
capability, its VRAM, the power limit the driver enforces and the PCIe link, and the row records them.

The only supported weights today are Unsloth's Qwen3.8-Flash-Next UD-IQ4_XS v3.1, converted by this repository:

| What | Size | Where |
|---|---|---|
| Unsloth's GGUF, 4 shards | 93,672,359,738 bytes | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| the pack it converts to | 90.1 GiB | built by `tools/convert/convert.py`; the commands are in the [README](../README.md#build-and-run) |

The GGUF and the pack need about 190 GiB of disk, and a run needs 64 GiB of RAM. On the reference machine the
conversion took 235 s and the first load 88 s; every load after that reads the pack.

Qwen3.8-27B from Unsloth's UD-Q4_K_XL is the second supported weight. Its GGUF converter path is not written yet, so
no 27B row can exist until it is.

## The cards

VRAM, SM count, bandwidth and power are vendor figures. The last column is what this repository's reference run
measured about the expert pool.

| Card | VRAM | SMs | Bandwidth GB/s | Power W | Experts |
|---|---|---|---|---|---|
| RTX 4090 | 24 | 128 | 1008 | 450 | 18.2 GiB in VRAM, 41.3 GiB in pinned RAM (measured) |
| RTX 6000 Ada | 48 | 142 | 960 | 300 | more of the pool stays in VRAM |
| L40S | 48 | 142 | 864 | 350 | more of the pool stays in VRAM |
| L40 | 48 | 142 | 864 | 300 | more of the pool stays in VRAM |
| RTX 5000 Ada | 32 | 100 | 576 | 250 | more of the pool stays in VRAM |
| RTX 4500 Ada | 24 | 60 | 432 | 210 | about the 4090's split |
| L4 | 24 | 58 | 300 | 72 | about the 4090's split |
| RTX 4000 Ada | 20 | 48 | 360 | 130 | part of the pool moves to pinned RAM |
| RTX 4080 SUPER | 16 | 80 | 736 | 320 | part of the pool moves to pinned RAM |
| RTX 4070 Ti SUPER | 16 | 66 | 672 | 285 | part of the pool moves to pinned RAM |
| RTX 4070 SUPER | 12 | 56 | 504 | 220 | part of the pool moves to pinned RAM |
| RTX 4070 | 12 | 46 | 504 | 200 | part of the pool moves to pinned RAM |

A card below 24 GiB still runs. The engine places as many experts as fit in VRAM and keeps the rest in page-locked host
RAM, so decode reads part of the weights over PCIe and runs slower than a card that holds the whole pool. A row from
such a card should say so in `notes`. The bench refuses a run only when experts would have to be read from the SSD,
because those numbers depend on the disk.

## Running it

```sh
git clone https://github.com/T-Crypt/halocline && cd halocline
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/convert/mtp_fetch.py fetch --out mtp-qfn
.venv/bin/python tools/convert/convert.py --gguf Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
    --no-vision --compat-bf16 --mtp-source mtp-qfn --out QFN-UD-IQ4_XS.halo

./bin/halocline-bench --community --model QFN-UD-IQ4_XS.halo --submitter YOUR_NAME
```

The bench starts its own engine three times, once for the decode fixtures and once per prompt depth, so every number is
a cold measurement. It prints the row, adds it to `docs/community/benchmarks.json`, then prints the hardware it measured
before the engine started. On the reference 4090 the whole preset takes about six minutes.

Run it on an idle machine. Each engine start pins about 47 GiB of experts in host RAM, and prefill falls when the host is
short of free memory: two runs of this preset an hour apart on the reference 4090 measured 1,958 and 5,070 t/s for the
same 64K prompt. The row records one number, so state in `notes` when the machine was busy.

`halocline-bench --hardware` prints the host figures alone, with no model and no engine start. Use it to check a new machine
before a run.

## What a row records

| Field | Meaning |
|---|---|
| `model`, `quant`, `kv` | what ran: the adapter's model name, the pack's quant label, the KV cache format the engine used |
| `commit`, `date`, `submitter` | the engine commit the binary was built from, the day, the name you gave |
| `gpus[].name`, `vram_gib`, `power_w` | the card, its VRAM, and the power limit the driver enforces (not the card's maximum) |
| `gpus[].cuda_cores`, `compute_cap` | as NVML counts them: 16,384 and "8.9" for a 4090 |
| `gpus[].pcie`, `h2d_gbps` | the link's maximum generation and current width, and a measured pinned host-to-device copy |
| `ram_gib`, `ram_gbps`, `cpu` | host RAM and its measured read speed, and the processor |
| `os`, `driver` | the system and the NVIDIA driver version |
| `prefill_16k_tps`, `prefill_64k_tps` | prompt tokens per second at each depth, one cold run each |
| `decode_tps` | generated tokens over decode time, summed over the three fixtures (not the mean of them) |
| `decode_{conv,struct,prose}_tps` | the same per fixture |
| `accept_*_pct`, `draft_len_*` | draft tokens accepted, and offered per window, when the pack has a drafter |
| `engine_args` | the flags the preset passed, without the model path |

## Comparing rows

Two rows compare when `model`, `quant` and `kv` match. The preset fixes the rest. Rows from different cards are what
the table is for: the same weights and the same protocol on different hardware.

The 12 GiB and 16 GiB cards keep part of the expert pool in host RAM, and decode reads it over PCIe. Their rates say
more about that link than about the engine's kernels, and `notes` should say the pool did not fit.

`docs/community/benchmarks-basalt.json` is Basalt's own table, frozen. Those runs are Blackwell cards with NVFP4 and
IQ3_S packs on an engine that is not this one; they are kept for the record and are not comparable with any row in this
file.

## Submitting

1. Run the command above. The card name, VRAM and power limit come from the driver, so do not edit them. Add `notes`
   (200 characters) when the run needs a caveat, such as experts in host RAM or a shared machine.
2. `python3 tools/check_benchmarks.py docs/community/benchmarks.json` must pass. It enforces the file's format, the
   field types and one row per setup.
3. Commit the file and open a pull request with the row. There is no CI on this repository: the maintainer runs the same
   checker before merging.

If the bench cannot write the file it prints the row and says where to paste it. A rerun on the same setup replaces that
setup's row in place and keeps its notes; change a run rather than adding a second row for it.
