# Precision

> [!WARNING]
> Carried over from Basalt's user docs with the names changed. Parts of this page describe code Halocline removed (multi-GPU, sm_120/Blackwell, Windows, DeepSeek) or packs it does not use (NVFP4). Halocline runs on one sm_89 card with Unsloth GGUF packs; [SPECS.md](../../SPECS.md) and [RESEARCH.md](../../RESEARCH.md) are current. This page is due for a rewrite.

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): stages, prefill, the measured costs, drafting and the lossy flags.
- [DeepSeek-V4.1-Flash](deepseek41.md): stages, the prompt path, verify windows and quality against its BF16 reference.
- [Qwen3.8-27B](qwen38-27b.md): the dense stack's roundings and quality against its BF16 reference.
- [How quality is measured](quality.md): the gate, the BF16 references, greedy identity and reproduction commands.
