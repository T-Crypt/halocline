# Halocline dense against NInfer with thinking off on both (2026-11-18)

2026-11-18 14:35 to 15:05 MST, aphotic, RTX 4090. `vs_ninfer_nothink.py`: the 2026-11-18 `vs_ninfer.py` with
`chat_template_kwargs: {"enable_thinking": false}` on both engines, through llama-swap, streamed, greedy, 1,200 tokens,
order Halocline, NInfer, NInfer, Halocline. Entries: `Halocline-27B-UD-Q4_K_XL` (release 75726a0f, K8V4, 175,616) and
`NInfer-Qwen3.8-27B` (its served flags). Why: with thinking on, Halocline wraps its thinking at 688 of 1,200 tokens and
answers in a second pass while NInfer thinks its own way, so the two wrote different kinds of text
(the lab's 2026-11-18 stream-path measurement: the engine's own timings had covered only the last pass, fixed in `serve/` ccc518c7). `reasoning_share` is 0.0 on every reply of both: both honoured the switch.

| workload | Halocline | NInfer | Halocline vs NInfer |
|---|---|---|---|
| short prose | 83.4, 83.3 | 92.9, 92.6 | **-10.1%** |
| short struct | 154.1, 154.1 | 155.3, 154.5 | -0.5% |
| 36K prose | 77.5, 77.4 | 85.7, 85.3 | **-9.4%** |
| 36K struct | 149.2, 147.9 | 137.5, 136.5 | +8.4% |
| 137K prose | 64.3, 63.6 | 70.8, 70.6 | **-9.5%** |
| 137K struct | 127.1, 125.8 | 116.1, 116.4 | +8.8% |

Read: on the same kind of text Halocline is behind NInfer by about 10% on prose at every depth, ahead by about 9% on
JSON at 36K and 137K, and level on short JSON. **The switch bar (standing order 6: every row at or above NInfer) is not
met: the three prose rows fail.** The earlier "ahead on 4 of 6" (thinking on) compared a wrapped-up
thinking reply against NInfer's own. The texts still differ between engines (each writes its own story), so a row moves
with what each wrote; prose against prose is the fair reading. Quality is unchanged by this: Halocline's depth dNLL is
about 10 times closer to BF16 (`../ninfer-depth-ppl`).

Where the prose gap is: Halocline's drafter accepts 56-58% of drafts on prose against 83-85% on JSON
(the same day's single-pass measurement with thinking off). NInfer drafts 3 MTP tokens with `--lm-head-draft`. The levers are lossless
by the verify rule (W15 draft head, draft policy on prose, the drafter's cache precision), per standing order 7.
