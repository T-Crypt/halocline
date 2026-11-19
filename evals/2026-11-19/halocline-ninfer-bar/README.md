# The NInfer switch bar on the served 262K profile, both engines in one session (2026-11-19)

Halocline roadmap standing order 6: at a served 262,144-token context, Halocline matches or beats NInfer's client-seen
decode on all six workloads, keeping its quality lead. Through llama-swap :9090, one engine on the card at a time
(Berth swaps them). `Halocline-27B-UD-Q4_K_XL` = release 5aec8e3a, `--spec 8 --long-context --vision cpu` (rotated 4-bit
K and V, 262,144). `NInfer-Qwen3.8-27B` = its served flags. Power limit 350 W. `run.sh`; `failed-race/` is a first start
that fired a second before the previous run's Berth lease was freed (both engines refused to start; no data).

## Speed (`vs_ninfer_nothink.py`, `speed.json`)

Thinking off on both (`reasoning_share` 0.0 on every reply), greedy, 1,200 tokens, streamed, client rate
(tokens - 1) / (last delta - first delta), order Halocline, NInfer, NInfer, Halocline.

| workload | Halocline t/s | NInfer t/s | Halocline vs NInfer |
|---|---|---|---|
| short prose | 81.6, 81.6 | 92.9, 92.5 | **-12.0%** |
| short JSON | 153.1, 151.0 | 155.2, 154.5 | **-1.8%** |
| 36K prose | 73.9, 73.9 | 85.6, 85.3 | **-13.5%** |
| 36K JSON | 151.0, 151.0 | 137.4, 136.4 | +10.3% |
| 137K prose | 64.3, 62.8 | 70.8, 70.6 | **-10.1%** |
| 137K JSON | 132.0, 128.7 | 116.0, 116.3 | +12.2% |

**The bar is not met: four of six rows lose** (three prose rows and short JSON). This is the first same-session
measurement at 262K; it agrees with the separate-session numbers in the roadmap (W22 against the 2026-11-18 NInfer
column) within 1 t/s.

Prompt time, read off the first token (`ttft_s`; NInfer's second arm reused its cached prompt, so its first arm is the
comparison): 36K 13.15 s against 11.37 s (Halocline 16% slower), 137K 83.6 s against 63.9 s (31% slower). The bar
counts decode only, but on the 262K profile the prompt side is behind too: the 4-bit cache prefills on the 4-warp
`prompt_v4` kernel (`../../2026-11-18/halocline-long-context`).

## Tool calls (`tools/tool_calls`, 24 cases, `tools-*.jsonl`)

Six tools offered every time; 12 cases that should call one (right tool, required arguments present), 12 that should
not. Thinking on as served, temperature 0, `max_tokens` 4096.

| | call | none | all |
|---|---|---|---|
| Halocline | 12/12 | 12/12 | 24/24 |
| NInfer | 12/12 | 12/12 | 24/24 |

The two engines decide all 24 cases alike. The set is too easy to separate them; it is a floor, not a ranking.

## What closes the decode gap (measured today, `../halocline-draft-head-prize`)

The frequency-ranked draft head gives prose +11-12% at short and 36K on the tiled-kernel build, which would put short
and 36K prose within about 1-3% of NInfer; it costs JSON 1-5% through over-drafting, which W35's policy fix targets.
The release candidate that lets it load at 262K is `../halocline-rc-aa156951`.
