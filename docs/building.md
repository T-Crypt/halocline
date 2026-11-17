# Building and testing

> [!WARNING]
> Carried over from Basalt's user docs with the names changed. Parts of this page describe code Halocline removed (multi-GPU, sm_120/Blackwell, Windows, DeepSeek) or packs it does not use (NVFP4). Halocline runs on one sm_89 card with Unsloth GGUF packs; [SPECS.md](../SPECS.md) and [RESEARCH.md](../RESEARCH.md) are current. This page is due for a rewrite.

The build commands are in the [README](../README.md). Windows: [platforms](platforms.md).

## What is in the repository

- `src/`, `include/`: the engine (`halo`, `halo-bench`, `halo-probe`, `halo-vision`).
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.basalt` reader (`model_file.py`), the quality gate (`ppl_gate.py`), the
  greedy-identity check and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

The build fetches nothing. The ggml and llama.cpp parts Halocline uses (quantization formats, CPU dot products, MMQ) are in `src/` and
`include/halo/ggml/`, under the MIT licence in `THIRD_PARTY_LICENSES`.

`-DWERROR=ON` makes project warnings errors (CI sets it on Linux).

## Building only some models

`-DHALO_MODELS=<ids>` is a semicolon list of architecture ids; the default builds every model. A model left out is not compiled at
all: its C++ and CUDA sources, its kernels and its tests are skipped, which is most of the build time for DeepSeek.

| Model | Id |
| --- | --- |
| Qwen3.8-Flash-Next | `qwen4exp` |
| DeepSeek-V4.1-Flash | `deepseek41` |
| Qwen3.8-27B | `qwen35` |

```sh
cmake -S . -B build -DHALO_MODELS=qwen4exp                  # Flash-Next only
cmake -S . -B build -DHALO_MODELS=deepseek41                # DeepSeek only
cmake -S . -B build -DHALO_MODELS=qwen35                    # the 27B only
cmake -S . -B build "-DHALO_MODELS=qwen4exp;deepseek41"     # two of them; quote the semicolon
```

An unknown id or an empty list stops the configure and names the valid ids. A pack of a model the build left out fails at load with
`architecture is 'deepseek41', a Halocline model this build left out (built with HALO_MODELS=qwen4exp); rebuild with cmake
-DHALO_MODELS="qwen4exp;deepseek41;qwen35" to load it`. The value is kept in the build folder's cache: to change it, pass the option again.

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`.

- Kernel parity tests need the GPU. `expert_parity` and `pool_test` use a synthetic fixture; `--file experts.bin` checks real blobs.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DHALO_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

`halo-bench` reads its prompts from `data/bench` (run from the repository root, or pass `--fixtures DIR`).
