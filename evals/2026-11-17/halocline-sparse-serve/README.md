# Halocline: first sparse serve on Linux, measured against the §11.7 floors, 2026-11-17 (the workstation, ticket the board)

- Build: `~/wt/halocline-import` `scope-wave1` at `d314a21a`. Pack: the interim compat pack
  (`../halocline-first-load/`: hc_* cast to BF16, no drafter, no vision), on `/data` (SATA).
- Server: `bin/basalt serve PACK --host 127.0.0.1 --port 8091 --alias qfn-halocline --spec 0 --prefill auto
  --max-context 262144 --kv int8` under a Berth foreground lease; the card otherwise idle, NInfer parked.
  Experts: 41.3 GiB page-locked in RAM, the rest in VRAM, nothing on the SSD; about 4 GiB of RAM left free.
- Client: `scripts/llama-swap/lsw.py smoke qfn-halocline --url http://127.0.0.1:8091 --long` (the floors' harness).
  Strata's standard-v1 client refuses a non-Strata server (`standard-v1-refused.txt`), so the depth rows come from
  the in-tree `basalt-bench` (below), a different protocol.

## Smoke against the floors (`smoke-1.txt`, `smoke-1.json`): every check passes

| Measure | Halocline (no drafter) | Floor (Strata 0.2.10 + DFlash2) |
|---|---|---|
| warm decode, short | 102.6-103.2 t/s (cold 97.9) | 86 (standard-v1 short), smoke 90.6-101.5 |
| 36K needle | decode 91.3 / prompt 6,274 t/s | decode 70 / prompt 2,400 |
| 137K long | decode 91.3 / prompt 6,523 t/s | decode 93 / prompt 2,860 |
| sampling, tool call | pass | pass |

The 137K prompt rate is over the 121,016 tokens not already cached (16,384 came from the 36K needle's prefix). 137K
decode is 2% under the floor; the floor has a drafter (DFlash2) and this pack has none, so drafting is the open lever.
One run; repeat before treating these as the engine's figures.

## basalt-bench depth rows (`basalt-bench-1.txt`)

`bin/basalt-bench --model PACK -t conv,struct,prose -p 16384,32768,131072 --seed 1 -- --spec 0 --prefill auto --kv int8
--max-context 262144`, its own engine process under a Berth lease (ready in 150 s). The bench prints "build c7132d5d62":
the hash CMake recorded at configure time; the binary is the `d314a21a` tree.

| test | Halocline | floor |
|---|---|---|
| decode conv / struct / prose | 104 / 106 / 106 t/s | 86 short (standard-v1) |
| prefill 16K | 5,243 t/s | 1,290 |
| prefill 32K | 6,475 t/s | 2,400 (36K) |
| prefill 131K | 6,354 t/s | 2,860 (137K) |

Not the same protocol as standard-v1 (different prompts and a cold process per depth), so read it as "above the floor
by a wide margin", not as a like-for-like ratio.
