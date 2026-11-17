# Contributing to Halocline

Halocline is one engine for one card and one model family: sm_89 (Ada) and the Qwen4 family. Every rule below exists to
keep the measurements honest and the engine fast on that target. The README states the scope; this file states the
rules for a change.

## Scope

- **CUDA on Ada (sm_89) only.** No ROCm, no Intel, no Apple, no CPU-only main path, no Blackwell, no Hopper. The build
  refuses any other compute capability, and pull requests adding another architecture are closed. The method for
  another card is in [`RESEARCH.md`](RESEARCH.md): fork it as `halocline-<arch>`.
- **The Qwen4 family only.** Another model family belongs in `halocline-<family>`, with its adapter under
  `src/models/<family>/`.
- **Linux x64.** Windows files are being removed; do not add platform branches to the engine. Platform differences go
  in the platform layer.
- **One fast path.** No per-user tuning flags and no alternative kernels picked at run time. A change that wins
  replaces the old path; a change that does not win does not merge. Diagnostic flags stay off by default and are not
  documented as settings.
- **No shims and no dead code.** A removed flag, option or code path is gone: no alias, no fallback, no "was retired"
  message. Code nothing uses is deleted in its own commit.
- **Defaults are measured.** A default, or an automatic pick that replaces one, changes only after a measured run on
  the reference machine (`SPECS.md`).
- **No environment-variable knobs.** Everything the engine reads is a command-line flag or a config entry. Diagnostics
  are flags too, off by default.
- **Decode comes first.** A prefill gain never costs decode speed, and every prefill change reports a decode number.
- **Engine log lines carry their level.** A line to stderr goes through `include/halo/log.hpp`
  (`halo::log::info`, `warn` or `error`), which writes `halo <area>: [warning: |error: ]<text>`. The server and the
  terminal read the level from that marker, never from the words. `tools/check_stderr_logging.sh` fails when engine
  code names stderr anywhere else. Lines are terse facts: what was decided, its value, and at most the flag to change
  it.

## Code structure

- **Shared code is model-agnostic.** It sees a model only through the adapter, a model's own code stays in
  `src/models/<model>`, and a change for one model leaves the others' output and speed as they were. See
  [`docs/adding-a-model.md`](docs/adding-a-model.md).
- **Logic exists once.** What two models need lives in shared code and takes each model's numbers as parameters.
- **One serving loop** runs every request of every model; a model plugs into it and never forks it.
- **The engine does the work, the server serves.** Tokenizing and per-request work run in C++. `serve/` parses, routes
  and streams, names no model, and never repairs model output; constrained decoding handles structure.
- **Build structure: no slow translation unit.** A clean Release build must finish in under 10 minutes on 32 threads,
  and nvcc compiles a `.cu` on one core, so the build takes as long as its slowest `.cu`. Keep every `.cu` to about a
  minute or two alone. A kernel instantiated in several variants gets one `.cu` per variant, behind a getter that
  returns its address. A `.cu` does not include another `.cu` (`tools/check_cu_includes.sh`; the exceptions are listed
  in `tools/cu_includes.allow`). Time a new or changed `.cu` alone and report it with the change.

## What every change must include

1. **A parity test** for every new or changed kernel: random inputs plus the real model geometry, compared against the
   current kernel it replaces and against the plain CPU reference (dequantize, then accumulate in double precision).
   Report the max absolute and relative error against both. A kernel meant to be bitwise-neutral must match the frozen
   output hashes of the current kernels. Output for a token must not depend on the window width or the token's
   position in it, and the test must check that.
2. **A profile.** A microbench line for the kernel: µs per call and achieved GB/s (or TFLOP/s) against the bytes or
   FLOPs it moves. An end-to-end number from `halo-bench`: ms per decode window and passes per second, not only tok/s,
   which drafting luck moves.
3. **A quality check:**
   - Bitwise-identical output: the greedy-identity check against the previous build, with the expert tiers pinned.
     Changes in module size move the auto-sized tiers and change rounding.
   - Changed numerics (new reduction order, precision, quantization): the KLD gate against the frozen BF16 base
     (`docs/precision/quality.md`), through the decode path. Report PPL, mean KLD and top-1 next to the previous
     build. Large changes need the full 100-chunk run, not the 20-chunk quick gate.
   - The `docs/precision/` page that covers the change is updated in the same change, with fresh numbers where it
     quotes them.
4. **A SASS record.** `cuobjdump -sass` hashes per object and per function under `gates/`. Kernels whose source did not
   change come out byte-identical, and inside a changed object only the functions the change targets may differ. A
   whole-object hash changes for reasons that have nothing to do with the code, so compare per function.
5. **The eval, if the change makes a claim.** Command, raw output, logs and the script go in `evals/<date>/<name>/`.
   Every number in the README, `RESEARCH.md` or a commit message cites the directory that produced it.
6. **Benchmark hygiene:**
   - Every performance number states the GPU power limit it was measured at.
   - A/B comparisons alternate the two builds in the same session (ABBA), with the expert tiers pinned to the same
     placement in both. Numbers from different sessions drift.
   - Numbers are cold first runs of the real engine, at a placement real users get. A gain that only exists with
     experts spilled to the SSD is not a headline.
   - A claim that something cannot get faster comes with a per-step breakdown against the measured hardware limit.

## Commits and docs

- One subsystem per commit, with the gate output in the message: build, SASS, CPU, GPU, and quality or identity when
  the change touches numerics.
- Commit messages state what changed and why. They carry no attribution trailers.
- Docs are append-only: when a fact changes, add the new state with its date. Do not delete the old line. "We tried
  that, here is what broke" is the most expensive knowledge in the repo.
- No number without a source. If a value comes from outside this repo, link it. If it was measured here, link the eval.
- Never write a secret, token or key into a file, a commit message or a log.

## Code style

Match the file you are editing. Names are spelled out in headers and tests, terse in kernels where the math is obvious.
Comments explain why a value is what it is, not what the next line does.
