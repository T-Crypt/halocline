# OrcaSAQ-2 27B: the image encoder in the borrowed prompt workspace (2026-11-21)

`after-long.sh orca` (branch `orca-vision-lend` = 0fe8224d + the lend and thread commits, the served profile: 262,144,
K8V4, `--mtp-kv int8 --mtp-window 32768`, `--spec 8`, `--vision dynamic`): `lsw.py smoke --long`, the three known-answer
images, then again with `hog.py` holding all but 198 MiB of the card.

| step | result |
|---|---|
| smoke | PASS; warm decode 122.7-123.5 t/s, needle 36K decode 112.3 t/s, long 137K 57.1 s, decode 96.5 t/s |
| after 137K | 3/3, 0.2-0.3 s an image |
| 198 MiB free | 3/3, 0.2 s an image |
| CPU fallbacks | 0; the engine log has `image encoder: in the prompt workspace (1292 MiB, the picture needs 966 MiB)` |
| engine peak before the hog | 23,198 MiB (`al-orca.mem.log.txt`; the 23,965 in `after-long.log.txt` includes the hog) |

Replies equal the 2026-11-20 `--vision cpu` reference (`../../2026-11-20/halocline-orcasaq-vision` in homelab). The
`after-long.log.txt` "fallback lines" count for orca read the dense run's engine log (the newest one at start); Orca's
own log (`~/.cache/halo/3130c06f81026089/engine.log`) has no fallback line after the lend line. The dense half of this
run, the ABBA and the gates are on branch `vision-pre-arena`, `evals/2026-11-21/halocline-vision-lend/`.
