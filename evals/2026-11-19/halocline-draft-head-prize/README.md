# The frequency-ranked draft head on the dense 262K cache: what it buys (2026-11-19)

W15's `--draft-vocab` (the MTP drafter scores a frequency-ranked subset of the 248,320-row head; FR-Spec, arXiv
2502.14856) had never been run for speed. It cannot load at 262,144 on the served build (53 MiB spare, the head copy
needs 131 MiB: `../../2026-11-18/halocline-draft-head-lc`), so every arm here runs `--long-context --max-context 245760`
(the same rotated 4-bit cache; the workloads reach 137K). Build `~/wt/halocline-w15` (draft-head-lc 30104ad4): the
tiled 4-bit decode kernel, before W22, so absolute rates sit below the served release; read the arms against each other.
Lossless: the verify pass scores every row (paired depth KLD +0.00000 in the 2026-11-18 directory).

`run.sh`, `decode.py` (the six vs_ninfer workloads, thinking off, greedy, 1,200 tokens, streamed, client rate), order
off v32k v64k v64k v32k off, a fresh server per arm, under `berth train`. Power limit 350 W. DeepSeek workers held E-cores
28-31 throughout.

| workload | full head | 32K head | 64K head | 32K vs full |
|---|---|---|---|---|
| short prose | 81.6, 81.5 | 91.5, 91.5 | 89.8, 89.6 | **+12.2%** |
| short JSON | 151.4, 152.2 | 149.2, 149.2 | 147.3, 147.4 | -1.7% |
| 36K prose | 68.8, 68.7 | 76.5, 76.5 | 74.5, 74.8 | **+11.3%** |
| 36K JSON | 132.6, 133.7 | 130.7, 131.9 | 132.5, 132.7 | -1.1% |
| 137K prose | 52.8, 52.7 | 53.8, 53.8 | 54.3, 54.2 | +2.0% |
| 137K JSON | 88.8, 89.1 | 84.2, 83.9 | 87.8, 88.1 | -5.5% |

Arms repeat to within 1.1 t/s. Drafts accepted / offered, short JSON: full 970/1104 and 977/1125, 32K 924/1303 and
924/1303, 64K 930/1289 and 926/1265 (per arm in `ab-*.json`).

## Read

- Prose gains 11-12% at short and 36K, the rows that fail the NInfer bar. At 137K the tiled attention kernel this build
  still has dominates the window, so a cheaper draft step moves little; the served W22 kernel leaves the drafter a
  larger share there.
- JSON loses 1-6% because the drafter offers 10-18% more drafts with either subset head and fewer are accepted: the
  window policy drafts wider. Suspected cause, not proven: the confidence `draft_end` reports is the subset's softmax,
  higher than the full head's (roadmap D5). Worker W35 is on it (`~/handoffs/halocline/2026-11-19-session/`).
- 32K beats 64K on prose, 64K holds JSON better at 137K. 32K is the arm to carry forward until the policy is fixed.
