# Halocline dense as served (2026-11-21)

What the dense path did for real work: the operator's coding agent (Pi, over the tailnet) on the served llama-swap
entry, `Halocline-27B-UD-Q4_K_XL` (UD-Q4_K_XL, `--spec 8 --long-context`, 262,144 tokens), release b2f81178, from 09:12
to 09:22. The numbers are llama-swap's activity log (`/api/metrics/activity`), whose token fields are the engine's own
`timings`. Local requests (benches, smoke checks) are left out. `rows.tsv` keeps counts and rates only: no addresses,
no prompt or reply text.

```sh
python3 served.py --url http://127.0.0.1:9090 --model Halocline-27B-UD-Q4_K_XL --since 2026-11-21T09:00 > summary.txt
```

| | served |
|---|---|
| requests | 59 (one cut off by an unload records no tokens), context 957 to 73,011 tokens |
| decode, replies of 200+ tokens | n=36: median 116.8 t/s, p25 110.5, p75 132.0, p95 156.2, max 166.4 |
| decode, every reply | n=58: median 115.8 t/s, p95 150.2 |
| prefill, fresh prompts (12,266 and 13,578 tokens, nothing cached) | 3,325 and 3,309 t/s |
| prefill, turns of 1,000+ new tokens over a warm cache | n=16: median 2,220 t/s, max 3,325 |
| draft tokens accepted | median 64%, best 80% |

## How it compares to the README's bench rows

These are not the same measurement as [`halocline-rc-aa156951`](../../2026-11-19/halocline-rc-aa156951/) and the speed
table in the README, and the two should not be subtracted:

- **The bench** is greedy, thinking off, 1,200 tokens of prose or JSON, timed by the client.
- **Served** is an agent's replies: code, tool calls and short answers (a median of 262 tokens, p75 607), timed by the engine
  (no HTTP or streaming time). Code is easier to draft than prose and harder than JSON, so the median sits between
  the bench's prose (68.7 to 91.7 t/s) and JSON (124.3 to 146.5 t/s) rows. The 166.4 t/s reply was 252 tokens at 16K
  context with 80% of its draft tokens accepted.
- **Prefill on agent turns** is mostly short chunks over a long cached prefix, which run below a full 36K prefill. The
  two fresh prompts in the window (12,266 and 13,578 tokens) ran at 3,325 and 3,309 t/s, in line with the bench's 3,383
  at 36K.

**Images.** This window was served with `--vision cpu`. Each request that brought a new screenshot spent 13 to 23 s
before its first byte, and that time is outside the engine's `prompt_ms`, so it is not in these rates. Since release
2321e84b the dense entry runs `--vision dynamic` with the encoder in the borrowed prompt workspace (0.53 s median,
[`halocline-vision-lend`](../halocline-vision-lend/)). The text path is the same, so the rates above still describe it.
