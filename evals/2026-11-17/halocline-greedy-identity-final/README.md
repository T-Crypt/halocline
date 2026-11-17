# Halocline: greedy identity, the pre-removal build against the published build, 2026-11-17

The §10.4 gate, rerun to cover everything after wave 2b: `901fd9c9` (Linux-only plus the single-GPU lock, before any
multi-GPU removal) against the published tree (rename, sm_89-only scope and docs).

- REF: `~/wt/halocline-import-901fd9c9/bin/basalt`, built 06:33 from `901fd9c9`.
- NEW: `~/wt/halocline-import/bin/halo`, built 10:20 from `scope-wave1` at the publish commit.
- Pack: `/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt` (the no-drafter pack was deleted 08:09).
- Command, under a Berth foreground pinned lease with the card idle:

  ```sh
  python3 tools/greedy_identity.py REF NEW --model PACK \
      -- --spec 0 --expert-cache 7000 --prefill auto
  ```

  The tool pins `--adapt-swaps 0` and `--max-context 8192` itself and runs the bench fixtures conv, struct and prose at
  300 tokens each with temperature 0.
- Placement is identical on both sides (`ref.err`, `new.err`): expert cache 7,961 slots, 18.23 GiB of VRAM, 41.30 GiB
  page-locked complement, 0 on SSD, the same profile (24,576 ranked pairs), slot 0 verified.

**Result: GREEDY IDENTICAL, 900 of 900 tokens (3 requests).** The scope removals, the rename and the sm_89-only build
changed no greedy token on one card.

Limits: greedy output only. It says nothing about sampling distributions; that is open work in `docs/testing.md`.
