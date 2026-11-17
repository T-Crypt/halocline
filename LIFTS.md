# Lifts: where the code came from

Halocline starts from one snapshot and has no upstream after it. Later releases of the projects below get read, and
anything taken from them is recorded here with the measurement that justified it. Licences and notices for every
source are in `LICENSE` and `THIRD_PARTY_LICENSES`.

## The snapshot

The first commit of this repository holds the tree of one merge, made 2026-11-16:

| Part | Commit | Notes |
|---|---|---|
| basalt_ada, branch `fork_4090` | `24362905a502d5d39875f29cc139b960086875aa` | 2026-11-15, Basalt 0.4.0 plus 32 commits of an Ada (sm_89) port |
| Basalt `v0.4.1` | `d3b75742967c7f9fd892c6894a2b9ac6e36e0cf3` | 2026-11-16 |
| Their merge base | `d7347724` | Basalt 0.4.0+4 |
| The merge | `7731c37a58cf676e56e80d837f9691338bc9a039` | two conflicts: `CMakeLists.txt` (both sides kept) and `src/prefill/fp8_gemm.cu` (the fork's Ada path first, then upstream's per-device opt-in) |

Neither project's history is in this repository. Every commit after the first is Halocline's, each with its gate
results in the message and its hash files under `gates/`.

## Lineage

| Project | Licence | Relation |
|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) | MIT | Basalt descends from it. Strata 0.2.10 (`af6224f3`) is the speed floor Halocline is measured against; no code taken from it after the snapshot. |
| [Basalt](https://github.com/jesdga95/basalt) | MIT | the engine the snapshot is built on |
| [basalt_ada](https://github.com/Ex3c4Def/basalt_ada) | MIT | the Ada kernels: GDN chain, gated-residual reads, routing, and the f16 forms of the FP8 and FP4 dense products |
| [Infernix](https://github.com/Wallawalla47/Infernix) | Apache-2.0 | tile layouts and epilogue arithmetic in the dense FP4 and FP8 prompt products, the GDN gating product and the int8 prompt attention |
| [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | the quantization formats, MMQ kernels and reference quantizers |

## Taken after the snapshot

Nothing yet.
