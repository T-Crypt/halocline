# `.halo` model file

One file holds what the engine, the server and the vision sidecar load: the source GGUF's tensors and metadata, the
tokenizer and chat template, and, when the pack carries them, the drafter and the vision projector. Flags, expert
placement and conversation state are not in it. The container comes from Basalt v0.4.1 (`d3b75742`); this page
describes what this tree reads and writes.

## Layout

- **Header.** The first 64 KiB hold a 64-byte `FileHeader` and then one 128-byte `SectionEntry` per section: a name of
  up to 63 bytes, the offset, the size, a SHA-256 and 16 reserved bytes. The directory lives in the header, so a pack
  holds at most 511 sections ([model_format.hpp](../include/halo/core/model_format.hpp)).
- **Magic.** A new pack carries `HALOCLMF`. `BASALTMF`, the magic of every file written before the rename, still loads;
  no new pack carries it ([model_file.cpp](../src/core/model_file.cpp)).
- **Version.** The engine reads 2 only. A version 1 file is refused with the migration's name
  ([migrations](../tools/migrations/README.md)).
- **Alignment and fill.** Every section starts at a multiple of 2 MiB and the bytes between sections are zero. Inside
  `gguf`, every tensor starts at a multiple of 4096 ([model_file.py](../tools/model_file.py)).
- **Checks on open.** The magic, the version, the section count, the file's length against the header's `file_bytes`,
  and every section's alignment and bounds. Padding and the reserved bytes are not read.
- **The extension decides nothing.** Nothing in the engine requires one; the programs that tell a pack from a raw GGUF
  accept `.halo` and `.basalt` ([cuda_parity.cu](../tests/vision/cuda_parity.cu),
  [ple_reader_test.cpp](../tests/ngram/ple_reader_test.cpp)).

## `meta`

`meta` is a JSON object, parsed by the reader's own parser (`src/core/model_file.cpp`). The keys the converters write:

| Key | Holds |
|---|---|
| `format`, `version` | `"halo"`, `2` |
| `name` | the pack's name, which the server reports |
| `arch` | the architecture id, written by the dense converter |
| `converter` | the tool and, on a dense pack, the route: `{"tool": "tools/convert/convert.py", "path": "gguf_qwen35"}` |
| `sources` | one record per input: `path`, `size` and `fingerprint_sha256_first_last_64mib_and_size` |
| `gguf_kv` | the source GGUF's metadata, sampling defaults (`general.sampling.*`) included |
| `tensors` | the tensor table, below |
| `license` | the license name, and the quantizer or the license text where the license requires it |
| `gguf` | `lossless`, `block_count`, and `renamed`: the MTP layer's source names to its `mtp.*` names |
| `mtp` | the dense pack's drafter record (`layers`, `source`); on a sparse pack the drafter's source hash and format |
| `vision` | `mmproj` and `mmproj_bytes` |
| `conversions` | with `--compat-bf16`: each converted tensor's types, method and measured error |
| `checkpoint`, `sections_from` | a checkpoint pack's config hash and methods, or the base pack a `--sections-from` build copied from |

`meta.converter` names the tool and the route, not a revision, a version or a host: nothing in the format records who
built a pack or from which commit. `meta.sources` describes each input by its size and the SHA-256 of its first and last
64 MiB, and `meta.gguf_kv` keeps the source's quantize keys (`quantize.imatrix.*` in the pack on disk).

`meta.tensors[name]` holds `type`, `type_id`, `shape`, `bytes`, `offset` and, on a dense GGUF pack, `source_sha256`.
It covers the `gguf` section; a sparse pack's dense pool has its own index in `dense.index`. The reader turns
`gguf_kv` and `tensors` into one virtual GGUF whose tensor byte ranges sit in the `gguf` section, so an
adapter reads a pack through the same interface it reads a GGUF with. An entry's offset is relative to the `gguf`
section unless the entry names another `section`; the reader checks that a tensor fits its section, not that two
tensors do not overlap. `meta.arch` is informational: the adapter is picked from `general.architecture` in `gguf_kv`
([registry.cpp](../src/model/registry.cpp)).

## Sections

| Section | Holds | Dense `qwen35` | Sparse `qwen4exp` |
|---|---|---|---|
| `meta` | the JSON above | yes | yes |
| `tokenizer/<file>` | `vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json` and, when the source GGUF carries `tokenizer.chat_template`, `chat_template.jinja` ([tokenizer.py](../tools/tokenizer.py)) | yes | yes |
| `gguf` | the source tensors in their GGUF bytes: every tensor of a dense pack, its drafter's included, and a sparse pack's quantized non-expert rows, whose float forms sit in `dense` | 866 tensors | 1,080 tensors |
| `dense`, `dense.index` | the dense weight pool and its text index (`# dense index v3`), which also names the rows the pool leaves in `gguf` | no | yes |
| `experts`, `experts.layout` | the routed experts' blobs and their text layout (`# experts v5` in the pack on disk; v6 for NVFP4 packs, v7 for a layer whose gate and up types differ) | no | yes |
| `mtp/dense`, `mtp/dense.index`, `mtp/experts`, `mtp/draft_vocab` | the sparse path's drafter files; `mtp/draft_vocab` optional | no | yes, without one in this pack |
| `hc8` | the 8-bit hyper-connection copies the sparse decoder reads | no | yes, required by that start |
| `vision/mmproj` | the projector GGUF, whole | yes | only with `--mmproj` |
| `vision/text_vocab` | a metadata-only GGUF of the text model's tokenizer | yes | only with `--mmproj` |
| `engram`, `dspark/gguf` | reserved, see below | no | no |

`dense`, `dense.index`, `experts`, `experts.layout`, `mtp/*` and `hc8` belong to the sparse path. A dense pack carries
none of them: its 64 main layers and its MTP layer are all tensors in `gguf`.

### Qwen3.8-27B pack

A dense `qwen35` pack. `Qwen3.8-27B-UD-Q4_K_XL-vision.halo`, read with `tools/model_file.py`:

| Section | Bytes |
|---|---|
| `tokenizer/chat_template.jinja` | 9,993 |
| `tokenizer/merges.txt` | 3,353,258 |
| `tokenizer/token_type.json` | 744,960 |
| `tokenizer/tokenizer.json` | 531 |
| `tokenizer/vocab.json` | 5,737,005 |
| `gguf` | 17,649,160,192 |
| `vision/mmproj` | 931,146,432 |
| `vision/text_vocab` | 10,945,408 |
| `meta` | 8,137,817 |

The file is 18,622,458,969 bytes. The sections sum to 18,609,235,596 and the 2 MiB alignment of each one accounts for
the rest. The 866 tensors in `gguf` are stored in the source GGUF's types (F32, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL
and IQ4_XS in this pack), 15 of them under `mtp.*`.

### Qwen3.8-Flash-Next pack

A sparse `qwen4exp` pack. `QFN-UD-IQ4_XS-v3.1-compat-mtp.halo` holds 15 sections: the same `meta` and `tokenizer/*`,
`gguf`, `dense`, `dense.index`, `experts`, `experts.layout`, `mtp/dense`, `mtp/dense.index`, `mtp/experts` and `hc8`. It
has no vision section because it was built without a projector, and no `mtp/draft_vocab`.

## Drafters

### The MTP layer

A dense pack stores the source GGUF's drafter layer as `mtp.*` tensors inside `gguf`, not as a section: `mtp.fc.weight`,
`mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight`, the attention and FFN
projections, and the norms ([gguf_qwen35.py](../tools/convert/gguf_qwen35.py)). The converter renames the last block's
`blk.<n>.*` and `blk.<n>.nextn.*` tensors; `meta.gguf.renamed` records both names and `meta.mtp` records where they
came from. The adapter reads them when `--spec` is 2 or more ([adapter.cpp](../src/models/qwen35/adapter.cpp)).

A sparse pack stores its drafter as sections: `mtp/dense` (the BF16 tensors from
[mtp_fetch.py](../tools/convert/mtp_fetch.py)), `mtp/dense.index`, `mtp/experts`, and `mtp/draft_vocab` when the head
may propose a token subset only. Its adapter turns `--spec` on when the first three are present
([adapter.cpp](../src/models/qwen4exp/adapter.cpp)).

### The DFlash 2 block drafter

A dense pack built with `--drafter` carries the block drafter as `dflash.*` tensors inside `gguf`: one
`dflash.fc.<j>.weight` per target layer, `dflash.hidden_norm.weight`, `dflash.norm.weight`, `dflash.blk.<l>.*` for its
attention, FFN and the two convolutions, and the selector's three tensors. The shape and the target layers are
`qwen35.dflash.*` keys in `gguf_kv`. The converter writes them ([checkpoint_qwen35.py](../tools/convert/checkpoint_qwen35.py))
and the qwen35 adapter reads them when they are present ([adapter.cpp](../src/models/qwen35/adapter.cpp)). No pack on
this machine carries them, so this path is read from the code only.

## Vision

`vision/mmproj` is the projector GGUF copied whole. The engine opens the section in the pack, and the server's vision
config names the pack so the sidecar opens the section itself ([modelfile.py](../serve/engine/modelfile.py)). A start
with `--vision` reads the same section in the engine. `vision/text_vocab` is a metadata-only GGUF of the text model's
tokenizer, written beside the projector ([pack_vision.py](../tools/convert/pack_vision.py)); nothing in this tree reads
it today.

## Tensor types

Most tensors keep the GGUF type the source stores: F32, BF16, Q8_0, Q4_K to Q6_K and the i-quants. The checkpoint
converter adds two of Halocline's own.

### `F8_E4M3` (tensor type 44)

One E4M3 code per value, each matrix's FP32 scale a separate F32 [1] tensor `<name>.scale`; the kernels read rows of a
multiple of 256 values. A BF16 output head becomes rows of E4M3 with one FP32 scale per row in `output.scale` (F32
[rows]).

### `NVFP4_PLANES` (tensor type 45)

An NVFP4 matrix of `rows` x `K` (rows a multiple of 128, `K` of 64) as two planes: `rows * K / 2` bytes of E2M1 codes,
byte `b` of a row holding values `2b` (low nibble) and `2b + 1` (high nibble), then `rows * K / 16` UE4M3 scales, one
per 16 values, in 512-byte tiles of 128 rows x 4 groups, row-block major. The matrix's FP32 scale is a separate F32 [1]
tensor `<name>.scale`. [nvfp4_planes.hpp](../include/halo/kernels/nvfp4_planes.hpp) is the engine's copy of the layout.

The dense stack refuses NVFP4 in ggml blocks (type 40) and names the re-conversion it needs
([meta.hpp](../src/models/qwen35/meta.hpp)).

## Integrity

- **Written.** Both writers put each section's SHA-256 in its directory entry ([model_file.py](../tools/model_file.py)).
- **Not checked on load.** `ModelFile::open` checks the magic, the version, the section count, truncation and each
  section's alignment and bounds. It never hashes a section, so a changed byte inside one loads
  ([model_file.cpp](../src/core/model_file.cpp)).
- **Checkable.** `python3 tools/model_file.py verify PACK.halo [--section NAME ...]` (2026-11-18) repeats the
  engine's header checks and hashes every section against its directory entry, one line per section, exit 1 on any
  mismatch; the 17.3 GiB dense vision pack verifies in about 12 s. For a dense GGUF pack,
  `python3 tools/convert/gguf_qwen35.py verify PACK.halo MODEL.gguf [--mmproj PROJECTOR.gguf]` re-reads the source and
  compares every tensor's SHA-256, `meta.gguf_kv`, the tokenizer sections (rebuilt from the same GGUF) and the vision
  sections ([gguf_qwen35.py](../tools/convert/gguf_qwen35.py)); `docs/building.md` has both commands. Converting the
  same GGUF twice gives byte-identical packs (`tools/convert/test_pack_integrity.py`).
- **No whole-file checksum.** `file_bytes` catches a truncated file only, and a file longer than its table is accepted.
  The 40 reserved bytes of the header and the 16 of each entry carry nothing.

## Reserved names

Two names in [model_format.hpp](../include/halo/core/model_format.hpp) have no reader and no producer in this tree.
They stay in the list as history:

- `engram`, a table in a section of its own. Its loader went with the DeepSeek model in `e8af75c6` (scope: remove
  DeepSeek-V4.1-Flash and the Windows build branches). A tensor entry can still name a `section` and `SideTableSpec` still
  has a `Kind::Engram`, but no adapter asks for either ([adapter.hpp](../include/halo/model/adapter.hpp)).
- `dspark/gguf`, a DSpark drafter GGUF stored whole. The same commit removed its reader.
  `tools/convert/convert.py --dspark --into` still exists but refuses a pack whose `meta.arch` is not `deepseek41`
  ([convert.py](../tools/convert/convert.py)), and nothing in this tree writes that architecture.

## Writing a pack

`tools/convert/convert.py` has one route per pack. `--help` lists every flag.

```sh
# a dense qwen35 GGUF: every tensor copied as stored
python3 tools/convert/convert.py --gguf MODEL.gguf --mmproj MMPROJ.gguf --out MODEL.halo

# a dense qwen35 NVFP4 checkpoint, with the DFlash 2 drafter if wanted
python3 tools/convert/convert.py --checkpoint DIR [--drafter DFLASH2] --out MODEL.halo

# a sparse qwen4exp GGUF, with the drafter fetched first
python3 tools/convert/mtp_fetch.py fetch --out DIR
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --mtp-source DIR --out MODEL.halo
```

Without a projector, pass `--no-vision`. `--sections-from BASE.halo` copies the drafter and vision sections of a pack
byte for byte and regenerates `vision/text_vocab`.
