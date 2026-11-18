# `.halo` file format changelog

The version is the `u32` after the magic in the file's header. The engine reads one version; the
[model file](model-file.md) page describes what this tree reads and writes. Newest first; an entry before 2026-11-18
records what the tree did at its date.

## 2026-11-18: what this tree reads

- **Read:** version 2, with either magic. Every section entry is checked against the file's bounds and alignment; a
  section no reader names is carried and ignored.
- **A dense `qwen35` pack** holds `meta`, `tokenizer/*`, `gguf` and, when it was built with a projector,
  `vision/mmproj` and `vision/text_vocab`. Its MTP layer is `mtp.*` tensors in `gguf`, and a DFlash 2 block drafter is
  `dflash.*` tensors there; neither is a section. It has no `dense`, `dense.index`, `experts`, `experts.layout`,
  `mtp/*` or `hc8` section.
- **A sparse `qwen4exp` pack** holds `dense`, `dense.index`, `experts`, `experts.layout`, `mtp/dense`,
  `mtp/dense.index`, `mtp/experts`, the optional `mtp/draft_vocab` and `hc8`, which that start requires, and, with a
  projector, `vision/mmproj`.
- **No longer produced or loaded:** `engram` and `dspark/gguf`. `e8af75c6` removed the DeepSeek model, and with it the
  only readers and producers of both. The names stay in `include/halo/core/model_format.hpp` as history;
  `tools/convert/convert.py --dspark --into` refuses every pack this tree builds.
- **Written but not read:** `vision/text_vocab`, a metadata-only GGUF of the text model's tokenizer, is written beside
  the projector by both dense converters. Nothing in this tree opens it.
- **Unchanged:** a section's SHA-256 is written and not checked on load; `ModelFile.verify` and the qwen35 converter's
  `verify` are the checks.

## 2026-11-17: Halocline's magic, the same version

- **New magic:** a file written from now on carries `HALOCLMF` where it carried `BASALTMF`, and its `hc8` table carries
  `HALOCLH8` where it carried `BASALTH8`. `tools/model_file.py` and `tools/convert/hc8.py` write the new ones.
- **Every old file still reads:** the C++ reader and the Python one accept both magics, for the file header and for the
  hc8 table, so the packs already on disk load without a rewrite. Only a file written from now on gets the new magic;
  nothing rewrites a pack in place.
- **No version change:** no section, field, alignment or byte offset moved, so the version stays 2.
- **No hash to fail:** a section's SHA-256 in the header covers the hc8 table's bytes as the writer produced them, and
  no reader checks a section's hash when it opens the file (`ModelFile::open` checks the magic, the version, the size
  and the section bounds; `tools/model_file.py`'s `verify` is the caller's). A legacy table therefore takes the same
  path as a new one.
- **Extension:** a new pack is named `.halo`. Nothing in the engine requires an extension; the programs that decide
  "pack or raw GGUF" (`tests/vision/cuda_parity`, `tests/ngram/ple_reader_test`) accept both `.halo` and `.basalt`.

## Version 2

The current version. The engine refuses version 1 files and names
[tools/migrations/halo_v1_to_v2.py](../tools/migrations/README.md), which writes a version 2 file from a version 1 file without a
download or any re-quantizing.

- **Contains:** `meta`, `tokenizer/*`, `dense`, `dense.index`, `experts`, `experts.layout`, `gguf`, `mtp/*`, `hc8`,
  `dspark/gguf`, `vision/*`.
- **Added `hc8`, now required:** the int8 copies of the hyper-connection matrices the decode reads, the bits the
  engine used to compute at every start. A start reads them as they are and skips reading the BF16 matrices to
  quantize them (about half a second).
- **Removed `profile`:** the expert profile decides which experts start in VRAM. It is placement policy that improves
  independently of the weights, so it moved to the engine (`data/profiles`), where a better profile needs no new file.
- **`experts.layout` v7, no version change:** a layer whose gate and up rows differ in type (Unsloth's UD-IQ1_S: IQ2_S gate
  and IQ2_XXS up on one layer) gives the up type as a sixth column under an `# experts v7` header. The converter writes v7
  only for such a model; every other pack keeps v5 (GGUF) or v6 (NVFP4). An engine from before v7 refuses a v7 pack and asks
  for an update.
- **Tensor type 45, `NVFP4_PLANES`, no version change:** NVFP4 as a code plane and 128 x 4-tiled UE4M3 scales (the TMA
  layout; [model file](model-file.md#nvfp4_planes-tensor-type-45)), the FP32 scale a separate `<name>.scale`. The Qwen3.8-27B
  pack's MLP and head use it; the engine refuses that pack's older ggml-block form.
- **Tensor type 44, `F8_E4M3`, no version change:** one E4M3 code per value with the matrix's FP32 scale as a separate
  `<name>.scale` tensor (F32 [1]), as NVIDIA's FP8 checkpoints store them. The Qwen3.8-27B pack's projections use it. An engine
  from before it refuses such a tensor by name ("the dense kernels read ...").
- **Added `dspark/gguf`, optional, no version change:** a DeepSeek pack's DSpark drafter. The engine ignores sections it
  does not name, so an engine from before it runs such a pack plain, and a pack without it runs plain on any engine.
- **Trimmed `meta`:** no `sampling` (the server reads the GGUF metadata's `general.sampling.*` keys), no `profile` or
  `hc8` records and no converter revision, provenance that changed whenever a section was replaced.

## Version 1

- **Contains:** `meta` (with `sampling`, `profile` and `hc8` records and the converter's git revision),
  `tokenizer/*`, `dense`, `dense.index`, `experts`, `experts.layout`, `gguf`, `mtp/*`, `profile`, optionally `hc8`,
  `vision/*`.
- The first single-file format: everything the engine, the server and the vision sidecar load. Without `hc8` the
  engine quantized the hyper-connection matrices at every start.
