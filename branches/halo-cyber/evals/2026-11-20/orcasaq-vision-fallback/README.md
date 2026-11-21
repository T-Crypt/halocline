# The dense image encoder falls back to the CPU when the card is full (2026-11-20)

`--vision dynamic` places the encoder's arena (about 966 MiB) on the card when a request brings pictures. Before this
change, a request whose arena did not fit was refused (`no device memory for the image encoder`): through llama-swap
that happened after a 137K-token smoke run with OrcaSAQ-2 at 262,144 tokens. Now the picture is encoded on the CPU by
the encoder `--vision cpu` uses (`vision::CpuEncoder`, bf16-rounded GEMM inputs, half the cores), and the engine log
says so.

`run4.sh`: OrcaSAQ-2 vision pack, the served profile (262,144, K8V4, `--spec 8`, `--vision dynamic`), a 137K-token prompt
with no picture before it (so no arena is held yet), then `hog.py` holds all but 198 MiB of the free memory, then the
three known-answer images.

| | Encoder | Per image | Replies |
|---|---|---|---|
| card has room (`../orcasaq-vision`, after load) | GPU | 0.2-0.4 s | 3/3 |
| 198 MiB free (`run.log.txt`, `hog.txt`) | CPU fallback, logged three times (`engine-fallback-lines.txt`) | 2.6-3.2 s | 3/3, the same text as `--vision cpu` |

`round3-arena-already-held/`: the same squeeze after a picture had already gone through. The arena from that first
request was still held, so the pictures ran on the card (0.2 s) with 198 MiB free: the fallback only runs when an
arena has to be made.

`hog.py` is a test helper: it holds the memory for this run only and frees it on exit.
