# Vision on both paths, on the CPU encoder (2026-11-19)

Both packs now carry the vision projector and both models answer images with `--vision cpu`: the `halocline-vision`
sidecar encodes on the CPU, opens no CUDA context and leaves the experts and the VRAM plan as they were. Release
5aec8e3a, served through llama-swap on the reference machine.

## Packs

- Dense: `Qwen3.8-27B-UD-Q4_K_XL-vision.halo`, the projector from Unsloth's `mmproj-BF16.gguf` at `4af51b35` (built
  2026-11-17).
- Sparse: `QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo`, the served command with Unsloth's `mmproj-BF16.gguf` at
  `814db05f` (the same revision as the text shards) in place of `--no-vision`. Every text section is byte-identical to the
  previous pack (`sections-old-vs-vision.txt`, by the per-section sha256 the converter stores; `compare_sections.py`).
  `hc8` and `meta` differ only in the format magic and the name.

## Known-answer images (`vision_check.py`)

Three PNGs drawn by the script, so a reply can be scored: two shapes to count, a line of text to read, a bar chart whose
tallest bar has a known label and colour. Greedy, thinking off, one request an image.

| case | dense | sparse |
|---|---|---|
| shapes: "3 red circles, 2 blue squares." | PASS | PASS (twice) |
| text: "HALOCLINE 4090" | PASS, 1.8 s | PASS, 2.3 s |
| bars: "C, green" | PASS, 2.5 s | PASS, 2.9 s |

The first request of each run includes llama-swap loading the engine (146 to 368 s); a warm image takes 2 to 3 s.

## RAM

The sidecar holds about 1.1 GiB resident (`sparse-ram-after-load.txt`). The sparse engine sizes its page-locked expert
tier from free RAM less `--ram-reserve-mib` (2,048 by default), and the sidecar was not in that plan: on the first
long smoke the system ran short during the 137K needle and the host's out-of-memory guard stopped the engine
(`sparse-smoke-long.txt`, `ram-trace-2.log`). With `--ram-reserve-mib 4096` every check passes
(`sparse-smoke-long-2.txt`). What that reserve costs at depth, and whether it is needed at all, is
`../halocline-sparse-abba`.
