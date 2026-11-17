# Halocline in llama-swap: smoke through :9090, 2026-11-17 (the workstation, ticket the board)

- Entry: `Halocline-QFN-UD-IQ4_XS` in `reference/llama-swap/config.yaml` (homelab `7942ef6a`), started by llama-swap
  under `berth run --as halocline --group llama-swap --priority foreground --min-turn 5m` with drain-post/drain-slots;
  Berth engine entry `halocline` (`6c578900`: whole_machine, foreground, min_turn 5m, `/health` on :8092).
- Server: release tree `~/halocline/releases/d314a21a`, pack `QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt` (MTP drafter),
  `--spec 8 --prefill auto --max-context 262144 --kv int8` (choice: `../halocline-kv-kld/`).
- Client: `scripts/llama-swap/lsw.py smoke Halocline-QFN-UD-IQ4_XS --long` (default URL, llama-swap :9090).

| Check | Result | Without drafter, :8091 direct (`../halocline-sparse-serve/`) | Floor |
|---|---|---|---|
| warm decode, short | 147.8-181.6 t/s (cold 170.2) | 102.6-103.2 | 86 |
| sampling | `391`, 48 tok, 0.9 s | PASS | |
| tool call | `{"city": "Oslo"}` | PASS | |
| needle 36K | 36,467 tok, 6.8 s, prompt 6,298 t/s, decode 123.1 t/s | 6,274 / 91.3 | 2,400 / 70 |
| needle 137K | 137,400 tok, 22.1 s, prompt 5,664 t/s, decode 121.0 t/s | 6,523 / 91.3 | 2,860 / 93 |

Every check passes (`smoke.txt`, `smoke.json`), and the 137K decode now clears its floor (it was 2% short without
the drafter). The 137K prompt rate is 13% lower than the no-drafter serve: the drafter's VRAM comes out of the expert
cache. "prompt 0 t/s" on the speed line is expected: prompts of 64 tokens or fewer go through decode windows
(`--short-read`), and Basalt reports `prompt_per_second: null` for them.

Harness fix in the same change: `lsw_smoke.py` crashed on that null (`max()` over `None`), so a null rate now counts
as 0, as a missing key already did.
