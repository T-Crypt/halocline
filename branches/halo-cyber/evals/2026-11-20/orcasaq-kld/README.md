# OrcaSAQ-2 27B: what Halocline's served KV costs against llama.cpp's, at depth (2026-11-20)

Same GGUF (`OrcaSAQ-2-27B-Uncensored.gguf`, sha256 `4ab6bdf8...`). The model exists only quantized, so the reference is
the most exact run of these weights: llama.cpp (the lab's pinned build with `tools/depth_kld/llama-kl-first.patch`) with
f16 K and V. Each base: a D-token prefix of wikitext-103 train, then 1,024 scored rows, 16 chunks (`run.sh`, `base-*.log`).

| Depth | Arm | Mean KLD | Top-1 | PPL (base) |
|---|---|---|---|---|
| 65,536 | llama.cpp `-ctk q8_0 -ctv q8_0` (served cyber KV before 2026-11-20) | 0.00041 | 98.99% | 7.1000 (7.0995) |
| 65,536 | Halocline `--kv int8` | 0.00088 | 98.65% | 7.1081 |
| 65,536 | Halocline `--kv k8v4` (served) | 0.00163 | 98.01% | 7.1109 |
| 140,288 | llama.cpp q8_0, `-ub 1024` | 0.00053 | 98.87% | 5.8871 (5.8858) |
| 140,288 | Halocline `--kv k8v4` (served) | 0.00208 | 97.93% | 5.8997 |

Limits:

- The reference shares llama.cpp's kernels with its q8_0 rows; Halocline's rows also carry two engines' arithmetic
  differences. No neutral BF16 reference exists for this model.
- 16 chunks x 1,024 rows per depth, one run per arm.
- 140,288 llama.cpp: `-ub 4096` (the 65,536 setting) needed a 5.3 GB compute buffer that does not fit
  (`llama-q8-140288-ub4096-oom.log` in the lab notes); `run2.sh` reran it at `-ub 1024`, as the served entry runs.
- The base at 140,288 ran with 48 of 65 layers on the card (f16 K/V 8.6 GiB); the layer split does not change the math.
