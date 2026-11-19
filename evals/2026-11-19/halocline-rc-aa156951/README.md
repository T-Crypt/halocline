# Release candidate aa156951: the gate (2026-11-19 09:51-11:20)

aa156951 = public main 5aec8e3a + the vision projector on a read-only mapping (34b0b201, b58118d8) + the `/monitor/`
redirect (cabeafaa) + the frequency-ranked draft head (`--draft-vocab`, 42e89d97 and its merge) + the planner charging
only the scratch the pack runs (4386fa37, so the draft head fits at 262,144). `gate.sh`, run under `berth train`,
power limit 350 W. `run.log` has every line below.

| step | result |
|---|---|
| GPU suite (`ctest -L gpu`) | 125 / 125 |
| dense depth KLD, `--long-context`, paired against the served kernel (`../../2026-11-18/halocline-long-context`) | +0.00000 at 65,536 and at 140,288; +0.00000 at 65,536 with `--draft-vocab` 32K (`paired.txt`) |
| dense loads at 262,144 with the draft head | yes (`ctx=262144` in every rc arm) |
| dense vision, CPU encoder, three known-answer images | 3 / 3 (`vision-dense.txt`) |
| sparse, NVMe pack, `--ram-reserve-mib 2048 --vision cpu`, `lsw.py smoke --long` | all PASS: 36K prompt 6,470 t/s, 137K prompt 5,602 t/s (`sparse-smoke.txt`) |
| sparse vision | 3 / 3 (`vision-sparse.txt`) |
| vision sidecar memory | 858 MiB file-backed pack mapping, 231 MiB anonymous (`sparse-sidecar-smaps.txt`) |
| sparse lowest MemAvailable through the smoke | 4,011 MiB (`sparse.ram.log`) |

## Dense speed: 5aec8e3a against aa156951 with `--draft-vocab` 32K, 262K, ABBA (`decode.py`, t/s)

| workload | 5aec8e3a | aa156951 + 32K | change |
|---|---|---|---|
| short prose | 81.7, 81.5 | 91.6, 91.7 | +12% |
| short JSON | 152.7, 149.5 | 146.6, 146.3 | -3% |
| 36K prose | 74.0, 74.0 | 81.3, 81.7 | +10% |
| 36K JSON | 146.6, 149.6 | 145.3, 141.7 | -3% |
| 137K prose | 65.2, 65.1 | 69.4, 68.0 | +5% |
| 137K JSON | 130.9, 130.9 | 123.0, 125.5 | -5% |

Output stays the model's own (the verify pass scores every row of the full head); the JSON loss is the window policy
reading the subset head's chances as surer than they are (`../halocline-draft-head-prize`). Both served entries moved
to this release later the same day.
