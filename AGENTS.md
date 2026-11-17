# AGENTS.md

Halocline is an inference engine for one card and one model family: NVIDIA Ada (sm_89) and the Qwen4 family. The C++
and CUDA engine is in `src/` and `include/`, each model's code under `src/models/<model>`, and the Python server with an
OpenAI-compatible API is in `serve/`.

## Rules

[CONTRIBUTING.md](CONTRIBUTING.md) is the project's rulebook and applies to agents in full: scope, model layering, the
gates every change carries, benchmark hygiene, and commits. The rules below are only for agents.

- **Read before answering.** Check the code, the docs and the eval for the number before stating a fact about
  Halocline. A plausible path, flag or model id that you did not read is the most common failure.
- **Cite the run.** Every number you report names the directory under `evals/` or the commit and hash file under
  `gates/` that produced it. If no run exists, say so and stop.
- **Design the end state first.** Read the similar code and reuse it. Do not propose a feature before testing the
  architecture you would add it to.
- **Flag odd numbers.** Sanity-check every result against size, speed and earlier numbers, and report anything odd with
  its suspected cause.
- **No amends, no force-pushes.** Make new commits.
- **Do not push or publish anywhere** (GitHub, Hugging Face, any remote) without the maintainer's approval.
- **Do not write tools nobody asked for.**
