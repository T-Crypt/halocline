# OrcaSAQ-2 27B with the stock Qwen3.8-27B projector: images at 262,144 tokens (2026-11-20)

OrcaRouter ships no vision projector. `convert.txt`: the pack is OrcaSAQ-2's GGUF plus Unsloth's Qwen3.8-27B
`mmproj-BF16.gguf` (931,146,432 bytes, sha256 `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`, the
same bytes as `main`'s UD-Q4_K_XL vision pack's projector section). `verify.txt`: the 866 model tensors match the GGUF,
0 mismatches.

`run.sh` serves it at the branch's profile (262,144 tokens, K8V4, `--spec 8`) and runs three images whose answers are
known: three red circles and two blue squares to count, the words "HALOCLINE 4090" to read, a bar chart whose tallest
bar is green and labelled C. Greedy, thinking off.

| Encoder | shapes | text | bars | per image |
|---|---|---|---|---|
| `--vision cpu` | "3 red circles, 2 blue squares" | "HALOCLINE 4090" | "C, green" | 1.9-2.6 s |
| `--vision dynamic` (on the card) | "3 red circles, 2 blue squares" | "HALOCLINE 4090" | "C, green" | 0.3-0.4 s |

`serve-dynamic.out`: `vision=gpu0`, 23,086 MiB on the card with the model loaded. Through llama-swap after a 137K-token
prompt, `dynamic` refused every image (HTTP 400, `no device memory for the image encoder (966 MiB)`: the long prompt's
working memory stays on the card, and there is no CPU fallback), so the served config uses `--vision cpu`, which passed
all three again after that prompt (1.8-2.4 s). Three images are a working check, not
a vision benchmark; the projector was trained with the stock model, not the abliterated one.

Logs are named `*.log.txt` here because the repository ignores `*.log` outside `logs/`; the scripts wrote them as `*.log`.
