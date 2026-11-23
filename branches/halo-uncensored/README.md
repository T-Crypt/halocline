# `branches/halo-uncensored/`

This branch's own files. `main` never writes here, so a merge from `main` cannot collide with them.

| Path | What |
|---|---|
| [`examples/hauhaucs-27b.json`](examples/hauhaucs-27b.json) | The server config: port 8098, name `Halocline-HauhauCS-27B`, rotated 4-bit K and V at 249,856 tokens, the image encoder on the card |
| [`examples/defiance-cyber-27b.json`](examples/defiance-cyber-27b.json) | The same pack on port 8099 as `Defiance-Cyber-27B`, with `--chat-template defiance-cyber-27b.jinja` (put the file next to the config) |
| [`evals/2026-11-22/gate/`](evals/2026-11-22/gate/) | KLD and dNLL against HauhauCS's Q8_K_P, short and at 65K and 140K |
| [`evals/2026-11-22/fit/`](evals/2026-11-22/fit/) | The context the card fits, the needle and the images at 249,856 |
| [`evals/2026-11-22/bar/`](evals/2026-11-22/bar/) | Speed and tool calls against NInfer on the same weights |

## Engine changes against `main`

Two.

**Quantized Gated DeltaNet gates.** Halocline's gating kernels read `ssm_alpha` and `ssm_beta` as BF16 rows. `main`
loads Unsloth's Q8_0 copies of those tensors by decoding them to fp32 and rounding each value to BF16 once, the one
rounding a GGUF pack's load makes (roadmap D2), and refuses any other type with
`blk.0.ssm_alpha.weight is not the expected type and size`. HauhauCS stores them as Q4_K. This branch decodes Q4_K,
Q5_K, Q6_K and IQ4_XS with the engine's own transcriptions of ggml's `dequantize_row_*` (`halo/artifact/dequant.hpp`)
and reads F32 as stored, then rounds to BF16 exactly as for Q8_0. Q8_0 still decodes inline, unchanged.

| File | Change |
|---|---|
| `src/models/qwen35/adapter.cpp` | `gate_block` gives each type's block size; `quant_pair_as_bf16` decodes by type; the planner counts the gates at BF16 size |

The pack on disk stays byte-identical to the GGUF (`gguf_qwen35.py verify`: 866 tensors, 0 mismatches). On `main`'s
UD-Q4_K_XL pack, 20 KLD chunks run twice on `main`'s release and twice on this build give KLD 0.005603, top-1 96.529%
and PPL 6.92350 all four times, with byte-identical per-row output.

**`--chat-template FILE`** (also on `halo-cyber`). The server renders prompts with the given Jinja file instead of the
pack's template.

| File | Change |
|---|---|
| `serve/config.py` | the flag, and `chat-template` as a config key whose path is relative to the config file |
| `serve/server.py` | reads the file at start |
| `serve/frontend.py` | with a custom template: `reasoning_effort` goes to the template as sent (a mode name, not only `low`/`medium`/`high`), other `chat_template_kwargs` pass through by name, a system message after the first turn stays in place |
| `serve/service.py`, `serve/handler.py` | the stock effort handling is skipped for a custom template |
| `serve/tests/test_custom_template.py` | six tests |

Without the flag nothing changes: the server suite passes as on `main`.
