# OrcaSAQ-2 Cyber 27B Uncensored: Halocline against llama.cpp, same file, same card (2026-11-20)

The question: on one RTX 4090, from the same GGUF bytes, does Halocline's dense path hold more context and decode
faster than stock llama.cpp?

## Setup

| | Halocline | llama.cpp |
|---|---|---|
| Build | branch `halo-cyber` (main 6ba99fcf plus the IQ4_XS gating loader) | the lab's pinned build, `/usr/local/bin/llama-server` |
| Weights | `OrcaSAQ-2-27B-Uncensored.halo`, every tensor as stored (`verify.txt`: 866 tensors, 0 mismatches) | `OrcaSAQ-2-27B-Uncensored.gguf`, sha256 `4ab6bdf8...` |
| Context | 262,144 | 188,416 (the most that fits with q8_0 K and V) |
| KV cache | K8V4: K int8, V rotated 4-bit; drafter cache int8, 32,768-cell ring | q8_0 K and V, flash attention |
| Drafting | the GGUF's MTP layer, `--spec 8`, 32,768-row draft head (verify scores all 248,320 rows) | the GGUF's MTP layer, `--spec-type draft-mtp --spec-draft-n-max 4` |

RTX 4090 at a 350 W limit (2,600 MHz clock lock), one engine on the card at a time, under one Berth training lease.
`vs.sh` starts each engine on port 8096 and runs `decode.py`: six workloads (short, 36K and 137K tokens of filler, then
a prose or a JSON task), greedy, thinking off, 1,200 tokens streamed, decode rate = (tokens - 1) / (last delta - first
delta). Then needles: a passphrase at a quarter of the depth.

## Results

| Workload | prompt tokens | Halocline tok/s | llama.cpp tok/s | Halocline vs llama.cpp |
|---|---|---|---|---|
| short prose | 39 | 99.7 | 84.5 | +18.0% |
| short JSON | 62 | 171.4 | 150.7 | +13.7% |
| 36K prose | 36,431 | 93.4 | 66.3 | +40.9% |
| 36K JSON | 36,454 | 158.2 | 119.3 | +32.6% |
| 137K prose | 137,364 | 77.2 | 42.1 | +83.4% |
| 137K JSON | 137,387 | 134.7 | 76.4 | +76.3% |

| Needle | Halocline | llama.cpp |
|---|---|---|
| 181,851 tokens | found, 83.6 s to the first token | found, 94.9 s |
| 252,398 tokens | found, 135.8 s | does not fit (188,416) |

Peak card memory: Halocline 23,178 MiB, llama.cpp 24,086 MiB (`*.gpu.csv`, 2 s samples).

`fit.log.txt`: int8 K and V at 262,144 needs 10,742 MiB of cache with 9,651 MiB free and is refused; K8V4 loads.
`fit-1-release.log.txt`: `main`'s loader refuses the file (`blk.0.ssm_alpha.weight is not the expected type and size`).

## Limits

- The V cache is 4-bit on Halocline and 8-bit on llama.cpp: the extra context is bought partly with V precision.
- The engines write different text (`text_sha`), so each rate is measured on its own output.
- llama.cpp reused the prose request's cached prompt on the JSON rows (0.6 and 1.0 s to the first token). Compare
  prompt time on the prose rows: 36K 10.8 s against 12.2 s, 137K 56.2 s against 57.6 s.
- One run per engine. No quality score: no KLD, no task benchmark. The needles are a retrieval check, not a quality one.

## The loader change does not move `main`'s path

`ud-identity.sh`: the UD-Q4_K_XL pack (Q8_0 gating tensors) on `main`'s release engine and on this branch's build,
`--spec 0` so no timing-dependent draft windows, greedy short prose and JSON, 256 tokens. Both write the same bytes
(`ud-identity.result`: `4f7c107f0c7cf6d8`, `fb104a7f1e1cdc3d` on both).

Logs are named `*.log.txt` here because the repository ignores `*.log` outside `logs/`; the scripts wrote them as `*.log`.
