# Models

A model is one `.halo` pack, mixture-of-experts or dense. The server reads the architecture from the pack and picks
the chat format, tool syntax and thinking switch.

Halocline runs the Qwen4 family only. Two paths exist:

|  | Qwen3.8-Flash-Next | Qwen3.8-27B |
|---|---|---|
| Architecture id | `qwen4exp` | `qwen35` |
| Shape | 48 layers (36 Gated DeltaNet, 12 Qwen Sparse Attention); 512 experts, 10 routed plus 1 shared | 64 layers (48 Gated DeltaNet, 16 gated attention); dense FFN |
| Context | 262,144 | 262,144 |
| Weights | Unsloth UD-IQ4_XS v3.1, converted to a pack | Unsloth UD-IQ4_XS or UD-Q4_K_XL, conversion pending |
| Drafting | MTP drafter, `--spec 8` | the GGUF's MTP layer, in the plan |
| `--kv` | int8 (measured best), fp16, k8v4, q4_0 | int8 (served), fp16, k8v4, q4_0 |
| Status | runs, gated, served through llama-swap | does not run yet |
| Guide | [Flash-Next](qwen38-flash-next.md) | [Qwen3.8-27B](qwen38-27b.md) |
| Precision | [Flash-Next](../precision/qwen38-flash-next.md) | [Qwen3.8-27B](../precision/qwen38-27b.md) |

The dense path is the plan of record for Qwen4-27B; the sparse path is what runs today. The ordered work, with accept
criteria, is in [`docs/roadmap.md`](../roadmap.md).

DeepSeek-V4.1-Flash and the multi-GPU paths were removed with the sm_89 scope; their pages are gone. Basalt's changelog
still records them.
