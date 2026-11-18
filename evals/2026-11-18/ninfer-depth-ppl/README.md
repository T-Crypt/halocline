# NInfer quality at depth: target-token NLL against the BF16 base (Halocline roadmap D7)

2026-11-18 07:26 to 08:29 MST, aphotic, RTX 4090. NInfer `ninfer-perplexity-v3-be68237c` on its served artifact
`qwen3_8_27b.v3.ninfer` (NInfer's own requantized format: `q4_g64_fp16`, `q5_g64_fp16`, `q8_g32_fp16` and bf16
tensors, 15.9 GiB). It is the engine and artifact behind llama-swap's `NInfer-Qwen3.8-27B`. Run under
`berth train --as ninfer-depth-ppl` (`run.sh`, log `run.log`). Halocline arms are the 2026-11-17 depth run
(`../../2026-11-17/halocline-d8-depth`, UD-Q4_K_XL, the same BF16 depth bases).

## Method

- **Same tokens.** `prep.py` reads each depth base's own token ids (16 chunks of depth + 1,025 ids, the last 1,024
  scored). It decodes them through Halocline's `/detokenize` (the same vocabulary) and re-tokenizes every stream; all 32
  chunks round-trip exactly (`prep.json`, `roundtrip: true`).
- **Same rows.** A 1,024-token filler in front of each chunk, `--context` = depth + 1,025 and `--stride 1024`, makes
  NInfer's window 2 score exactly the base's 1,024 rows. `analyze.py` asserts `input_begin == 1024` and
  `scored_tokens == 1024` for every stream.
- **Metric: target-token dNLL**, mean NLL of the observed next token minus the BF16 model's NLL of the same token, per
  chunk, then the mean over 16 chunks with a 95% t interval; the last column pairs each engine against Halocline K8V4
  per chunk. **This is not KLD.** NInfer exposes no logits, so KLD is not possible; over- and under-confidence can
  cancel in dNLL, so a small dNLL is necessary for closeness to BF16, not sufficient. Lower is closer.
- Arms: NInfer served KV (`rk4v4-e8`) at 65,536 and 140,288; NInfer `int8` KV at 65,536, which separates NInfer's
  weight conversion from its cache.

## Result (`results.md`, from `python3 analyze.py`)

| Depth | Engine, KV | PPL / base | dNLL vs BF16 [95% CI] | vs Halocline K8V4, paired [95% CI] |
|---|---|---|---|---|
| 65,536 | Halocline, int8 | 7.0469 / 7.0407 | +0.00088 [-0.00088, +0.00263] | -0.00088 [-0.00190, +0.00014] |
| 65,536 | Halocline, K8V4 (served) | 7.0531 / 7.0407 | +0.00176 [-0.00037, +0.00388] | (reference) |
| 65,536 | Halocline, rotated 4-bit K and V | 7.0502 / 7.0407 | +0.00136 [-0.00097, +0.00368] | -0.00040 [-0.00177, +0.00097] |
| 65,536 | NInfer, rk4v4-e8 (served) | 7.1822 / 7.0407 | +0.01989 [+0.01570, +0.02409] | +0.01814 [+0.01462, +0.02165] |
| 65,536 | NInfer, int8 | 7.1626 / 7.0407 | +0.01717 [+0.01373, +0.02061] | +0.01541 [+0.01162, +0.01920] |
| 140,288 | Halocline, int8 | 5.8398 / 5.8356 | +0.00073 [-0.00119, +0.00264] | -0.00123 [-0.00203, -0.00044] |
| 140,288 | Halocline, K8V4 (served) | 5.8471 / 5.8356 | +0.00196 [+0.00008, +0.00384] | (reference) |
| 140,288 | NInfer, rk4v4-e8 (served) | 5.9542 / 5.8356 | +0.02013 [+0.01453, +0.02573] | +0.01817 [+0.01272, +0.02362] |

Read:

- Served against served, NInfer is about 11 times Halocline's distance from BF16 at 65K (+0.0199 against +0.0018) and
  10 times at 140K (+0.0201 against +0.0020). Every NInfer interval excludes zero and excludes Halocline's.
- **Most of NInfer's loss is its weights, not its cache.** With int8 KV NInfer is still +0.0172 at 65K; its 4-bit
  `rk4v4-e8` cache adds +0.0027 on top (about 14% of its served dNLL). Halocline repacks Unsloth's UD-Q4_K_XL tensors
  as stored, so its weights add no conversion step.
- NInfer's distance does not grow from 65K to 140K (+0.0199, +0.0201): its depth handling is not the problem.

## Files

`prep.py` (corpus builder; needs a Halocline server for `/detokenize`), `run.sh` (arms), `analyze.py` (tables),
`results.md`, `d65536/`, `d140288/` (`prep.json`, `manifest.json`; the `data/` text is regenerable by `prep.py` and is
not committed), one folder per arm with NInfer's `report.json`, its `.out`/`.err`, and `failed-0725/` (a first attempt
whose manifest lacked `modes`, kept).
