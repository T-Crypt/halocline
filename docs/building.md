# Building and testing

Halocline builds on Linux for one compute capability: sm_89, the Ada cards. The build fetches nothing; the ggml and
llama.cpp parts it uses (quantization formats, CPU dot products, MMQ) are in `src/` and `include/halo/ggml/`, under the
MIT licence in `THIRD_PARTY_LICENSES`.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build                       # bin/halocline, bin/halocline-bench, bin/halocline-probe, bin/halocline-vision
                                     # (bin/halo is a symlink to bin/halocline, for scripts written before the rename)
```

`-DWERROR=ON` makes project warnings errors. A clean build of both models takes about 25 minutes on 24 threads; most of
it is device code.

## What is in the repository

- `src/`, `include/`: the engine.
- `serve/`: the Python server with the OpenAI and Anthropic API and the stats page.
- `tools/`: the converter (`tools/convert/convert.py`), the `.halo` reader (`model_file.py`), the quality gate
  (`ppl_gate.py`), the greedy-identity check and the tokenizer.
- `tests/`: parity and unit tests. `data/`: expert profiles (`data/profiles`) and benchmark fixtures (`data/bench`).

## Building only some models

`-DHALO_MODELS=<ids>` is a semicolon list of architecture ids; the default builds both. A model left out is not compiled
at all: its C++ and CUDA sources, its kernels and its tests are skipped.

| Model | Id |
|---|---|
| Qwen3.8-Flash-Next | `qwen4exp` |
| Qwen3.8-27B | `qwen35` |

```sh
cmake -S . -B build -DHALO_MODELS=qwen4exp     # Flash-Next only
cmake -S . -B build -DHALO_MODELS=qwen35       # the 27B only
```

An unknown id or an empty list stops the configure and names the valid ids. A pack of a model the build left out fails at
load with the architecture and the ids the build has. The value is kept in the build folder's cache: to change it, pass
the option again.

## Tests

Configure with `-DBUILD_TESTING=ON`, build, and run `ctest --test-dir build`. The suites and what each proves are in
[testing.md](testing.md).

- The CPU suite runs anywhere. Give it a `TMPDIR` with 4 GiB free, not `/tmp` on a small tmpfs.
- The GPU suite needs the card alone. It is the parity check for every kernel.
- Server tests: `python -m unittest discover -s serve/tests -t .` after `pip install -r requirements-test.txt`.
- Tokenizer parity: register `tokenizer_parity` with `-DHALO_TOKENIZER_DIR=<a pack's tokenizer/ directory>`.

## Benchmark

`halocline-bench` reads its prompts from `data/bench` (run it from the repository root, or pass `--fixtures DIR`). The
standard benchmark and the fixed preset that fills the community table are in [community.md](community.md):

```sh
./bin/halocline-bench --model PACK.halo -- --kv int8 --prefill auto     # the standard protocol
./bin/halocline-bench --community --model PACK.halo --submitter NAME    # the comparable row
./bin/halocline-bench --hardware                                          # host figures alone
```

The quality gate is the same binary with `--quality N --kld-base FILE`; the method is in
[precision/quality.md](precision/quality.md).
