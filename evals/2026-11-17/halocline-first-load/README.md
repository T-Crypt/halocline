# Halocline: first Linux model load, 2026-11-17 (the workstation, RTX 4090, ticket the board)

The first model ever loaded by this engine on Linux (BUILD_SPEC §13: retire "nothing loads a model on Linux" first).

- Build: `~/wt/halocline-import` `scope-wave1` at `d314a21a` (wave 2b done), `build-89` (Release, sm_89).
- Pack: `/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat.basalt`, 89.29 GiB, 12 sections, built in 235 s by
  `tools/convert/convert.py --gguf <QFN UD-IQ4_XS v3.1 shard 1> --no-vision --compat-bf16` (`convert.log`). No drafter,
  no vision. **Interim artifact:** `--compat-bf16` casts the 195 `hc_*` tensors (Q8_0 in the Unsloth GGUF) to BF16,
  which the engine requires before it makes its own hc8 decode copies. That breaks "bytes unchanged" for those tensors
  (maintainer decision pending, see the HANDOFF block); every expert and quantized projection is copied as shipped.
  Fine for build-against-build gates; not evidence for the quality gate.
- Run: `bin/basalt --model PACK --tokens-file data/bench/conv.ids --max-new 64 --spec 2 --prefill auto --stats` under a
  Berth foreground lease (card otherwise idle). The adapter refuses to run without `--spec T >= 2` and `--prefill`.

| Measure | Value |
|---|---|
| Expert placement | VRAM 18.2 GiB (32%), RAM 38.0 GiB page-locked (68%), SSD 0 |
| RAM copy | 41.31 GiB read in 87 s (0.51 GB/s, unbuffered from SATA) |
| Time to first token | 88.3 s (dominated by the RAM copy) |
| Prefill | 37 tokens, 127 tok/s |
| Decode | 64 tokens, 50.8 tok/s (greedy, `--spec 2`, no drafter, 85.6% VRAM expert hits) |

Output (`text.txt`): a correct, coherent start of the requested B-tree vs LSM explanation. Decode is a smoke number,
not a floor comparison: no drafter (Strata's floor uses DFlash2), a 64-token run, the pack on SATA.
