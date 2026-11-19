# Precision

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): the sparse path, its KLD against BF16, the KV choice and the speed numbers.
- [Qwen3.8-27B](qwen38-27b.md): the dense path's reference, gate and KV formats, measured against BF16 at 36,864 tokens and at depth.
- [How quality is measured](quality.md): the gate, the BF16 references, greedy identity and reproduction commands.

DeepSeek-V4.1-Flash was removed with the sm_89 scope; Basalt's changelog still records it.
