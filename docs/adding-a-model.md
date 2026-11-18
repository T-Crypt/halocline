# Adding a model

A model is an adapter under `src/models/<arch>/` (headers under `include/halo/models/<arch>/`) plus a `Dialect` in
`serve/models/`. Shared code sees a model only through `ModelAdapter` and `ModelSession`
([adapter.hpp](../include/halo/model/adapter.hpp)); `tools/check_model_layering.sh` fails if shared code includes a
model or one model includes another.

## The two models

| Architecture | Source | What runs it |
|---|---|---|
| `qwen35` | `src/models/qwen35/` (adapter, metadata) | the shared dense stack: the adapter reads the pack into `dense::Geometry` and `dense::Weights`, and `src/dense/` runs the layers, the drafters, images and the session |
| `qwen4exp` | `src/models/qwen4exp/` (adapter, load, decode, prefill, serve) | its own start and serving runtime: `load()` returns `NotImplemented` and `run()` hands over to `run_engine()`, whose `QfnRuntime` derives from `program::serve::Runtime` |

Both register `halo::models::<arch>::make_adapter()` in the library `model_<arch>`. A dense model of GDN and gated
attention layers needs no kernels of its own; it needs a pack reader that fills the shared geometry and weights, as
[qwen35/meta.hpp](../src/models/qwen35/meta.hpp) and [qwen35/adapter.cpp](../src/models/qwen35/adapter.cpp) do.

## Registering it

1. Add the architecture id to `HALO_BUILTIN_MODELS` in [cmake/models.cmake](../cmake/models.cmake). `HALO_MODELS`
   defaults to that whole list, and `-DHALO_MODELS="qwen35"` builds a subset; a model left out is not compiled, linked
   or registered.
2. Put everything of the model in `CMakeLists.txt` under `if(HALO_MODEL_<ARCH>)`.
3. Nothing else. [builtin.cpp](../src/models/builtin.cpp) expands one registration per built model at start, so linking
   order decides nothing. `unavailable_architecture()` answers a pack this build left out with the models it holds and
   the cmake line that adds the missing one ([registry.cpp](../src/model/registry.cpp)).

The adapter is chosen from `general.architecture` in the pack's metadata, never from `meta.arch` or a file name
([registry.cpp](../src/model/registry.cpp)).

## The adapter

Required calls:

| Call | Returns |
|---|---|
| `arch()` | the id `general.architecture` must carry |
| `validate(const GgufFile&)` | an empty string, or why the file is not this model |
| `describe(const GgufFile&)` | `ModelDesc`: layers, width, vocabulary, context, and the expert counts where the model has experts |
| `load(const LoadRequest&, LoadedModel&)` | the loaded model, or the reason it cannot load; a model with its own start leaves this `NotImplemented` |
| `tokenizer_spec()` | the metadata key holding the token list and the server's chat format name |
| `token_names()` | the names of the end-of-turn, turn, image-pad and video-pad tokens, looked up in the pack's vocabulary |
| `pretokenizer()` | the pre-tokenizer's name in `tokenizer.json` and its split rule |
| `tool_form()` | the tool-call grammar |

Optional calls and what each one buys:

- `parse_flag`, `finish_flags` and `make_options`: the model's own flags and its options object, which derives from
  `CommonOptions`. A flag the model does not take falls through to the ordinary unknown-argument error.
- `planned_spec`: the window `--spec` defaults to for this pack. The server's start reads it from `--gpu-plan` and
  writes it into the model's entry ([config.py](../serve/config.py)).
- `quality_flags` and `quality_window`: how the quality gate starts the model and which window it may score.
- `model_name`: the name a community benchmark row records.
- `side_tables`: a per-token table read from storage. `qwen4exp` returns its PLE table; no adapter returns an Engram
  table.
- `shipped_profiles`: the `data/profiles` files the start expects for the pack. Neither shipped model overrides it.
- `run`: a model that owns its start and serving loop runs them here and returns an exit code.

## The session

`ModelSession` is what the server drives. A model fills the calls it supports; the rest keep their default
`NotImplemented` answers.

- **Requests:** `decode_window` and `prefill_chunk`; `max_window` (the rows a window may hold), `decode_batch` and
  `batch_rows` for batched serving, `prepare` before the first request.
- **Drafting:** `draft_begin` and `draft_end` for a drafter, `max_window` for the window it fills, `copy_drafts` when
  prompt-lookup drafts may stand in for the drafter's.
- **State:** `running_state` and `running_restored` for a checkpoint, with `running_bytes`; `save_cells`,
  `restore_cells`, `cells_bytes` and `parks_cells` for the conversation cache.
- **Memory:** `set_low_memory` and `set_vram_reserve` give VRAM back.
- **Images:** `encode_image` and `prefill_embeddings` for a model with its own encoder, `encode_patches` and
  `reads_patches` for one that reads the shared preprocessor's patches, `images_done` to release the encoder.
- **Reports:** `info_words`, `expert_counts`, `copy_logits` and `row_logprobs`, `disk_min_tokens`, and `probe_env` for
  `PROBE rates`.

## The pack it reads

The adapter is handed a `GgufFile` view of the pack: `meta.gguf_kv` as metadata and `meta.tensors` as tensor entries
over the `gguf` section ([model_file.cpp](../src/core/model_file.cpp)). A dense `qwen35` pack needs only the
`qwen35.*` keys and `tokenizer.ggml.tokens`; [model-file.md](model-file.md) lists them. A sparse model reads its own
sections through `core::model_sources()`.

## The server dialect

`serve/models/<arch>.py` subclasses `Dialect`: the chat template, the output parser, the stop texts, the image and
video markers, and the template capabilities it reports. [serve/models/__init__.py](../serve/models/__init__.py)'s
`dialect_for()` names the architectures it can serve and refuses any other, so a new architecture adds its line there.
Both Halocline models use the shared Qwen dialect ([qwen.py](../serve/models/qwen.py)).

## Fixtures and the quality gate

Decode fixtures are token-id files in `data/bench` (`conv.ids`, `struct.ids`, `prose.ids`, `prefill.ids`).
`halocline-bench` and `halocline-probe` look first for a directory named after the planned architecture under that path
([engine_session.cpp](../src/program/engine_session.cpp)), so a model can keep its own. Make them once with the model's
tokenizer and chat template; `halocline-bench --help` prints the recipe.

`--quality N` starts the model with the adapter's `quality_flags()`, caps the verify window at `quality_window()`, and
needs `--kld-base FILE` ([bench_main.cpp](../src/program/bench_main.cpp)).

## Tokens and kernels

The token id flags (`--eos-ids`, `--turn-token`, `--image-pad-id`) default from `token_names()` looked up in the
pack's `tokenizer/vocab.json` ([startup.cpp](../src/program/cli/startup.cpp)); the video pad has no flag and always
comes from the pack.

Shared kernels stay in `src/kernels/`; a model's own live under `src/models/<arch>/kernels/` and are built under its
cmake guard. A new kernel carries a parity test and a microbench line, as [CONTRIBUTING.md](../CONTRIBUTING.md) requires.
