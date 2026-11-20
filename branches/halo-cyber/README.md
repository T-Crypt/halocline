# `branches/halo-cyber/`

This branch's own files. `main` never writes here, so a merge from `main` cannot collide with them.

| Path | What |
|---|---|
| [`examples/orcasaq-27b.json`](examples/orcasaq-27b.json) | The server config: port 8096, name `Halocline-OrcaSAQ-27B`, K8V4 KV at 262,144 tokens |
| [`evals/`](evals/) | The runs behind the numbers in the branch README |

## Engine change against `main`

One, in the dense loader. Halocline's Gated DeltaNet gating kernels read `ssm_alpha` and `ssm_beta` as BF16 rows.
`main` loads Unsloth's Q8_0 copies of those two tensors by decoding them to fp32 and rounding each value to BF16 once,
the one rounding a GGUF pack's load makes (roadmap D2). OrcaRouter stores them as IQ4_XS, which `main` refuses with
`blk.0.ssm_alpha.weight is not the expected type and size`. This branch decodes IQ4_XS the same way, with the engine's
own transcription of ggml's `dequantize_row_iq4_xs`, then rounds to BF16 exactly as for Q8_0.

| File | Change |
|---|---|
| `src/models/qwen35/meta.hpp` | the IQ4_XS type id (23) |
| `src/models/qwen35/adapter.cpp` | `quant_pair_as_bf16` takes Q8_0 or IQ4_XS; the planner counts IQ4_XS gating tensors at BF16 size |

The pack on disk stays byte-identical to the GGUF (`gguf_qwen35.py verify`: 866 tensors, 0 mismatches). Q8_0 values
go through the same arithmetic as before. Checked on the UD-Q4_K_XL pack, `--spec 0`, greedy prose and JSON: `main`'s
release and this build write byte-identical text ([`ud-identity.result`](evals/2026-11-20/orcasaq-vs-llamacpp/ud-identity.result)).
