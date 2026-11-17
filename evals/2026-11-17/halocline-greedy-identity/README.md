# Halocline: the §10.4 greedy identity gate, 901fd9c9 vs wave 2b, 2026-11-17 (the workstation, ticket the board)

The gate owed since the scope removals began (BUILD_SPEC §10.4): the build before any multi-GPU removal against the
build after all of wave 2b, on one pack, greedy, placement pinned.

- REF: `~/wt/halocline-import-901fd9c9` (`901fd9c9`: Linux-only + the single-GPU lock, before wave 2b), Release sm_89.
- NEW: `~/wt/halocline-import` `scope-wave1` at `d314a21a` (wave 2b done: 2b1 to 2b14 and the serve test fix).
- Pack: `/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat.basalt` (see `../halocline-first-load/`).
- Command (under a Berth foreground lease, card otherwise idle):
  `tools/greedy_identity.py REF NEW --model PACK -- --spec 0 --expert-cache 7000 --prefill auto`
  (the tool's own pins: `--adapt-swaps 0`, `--max-context 8192`; the bench fixtures conv, struct, prose, 300 tokens each,
  temperature 0). `--spec 0` because the pack has no draft layer (serve refuses `--spec 7` without one).
- Placement: both builds report 7,961 experts in the GPU cache, 41.30 GiB page-locked in RAM, 0 on the SSD
  (`ref.err`, `new.err`). `--expert-cache 7000` did not set that count (it is not the slot total here); since both
  sides placed identically the gate holds, and what the flag means for this model is a follow-up.

**Result: GREEDY IDENTICAL, 900 of 900 tokens (3 requests).** Wave 2b removed the multi-GPU code without changing a
single greedy token on one card.
