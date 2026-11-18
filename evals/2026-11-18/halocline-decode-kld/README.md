# Depth-KLD gate for decode-kernel changes (2026-11-18)

The maintainer's rule for a decode attention change (2026-11-18): it may change summation order only if the paired depth
KLD at 65,536 and 140,288 is no worse than K8V4 today. `kld.sh LABEL BUILD` runs `halocline-bench --quality 16` against
the BF16 depth bases (`tools/depth_kld`; the protocol of `../../2026-11-17/halocline-d8-depth`), served K8V4 flags,
verify windows of 3 rows (`--quality-t` default, so the decode kernel scores every row), and pairs each arm per chunk
against the 2026-11-17 K8V4 rows with `kv_paired.py` (`paired.txt`, 95% t interval over 16 chunks).

| Arm | Build | 65,536: KLD / top-1 / PPL, paired vs reference | 140,288 |
|---|---|---|---|
| rel (calibration) | release b22ecbb2 | 0.00490 / 96.75 / 7.0531, +0.00000 [+0.00000, +0.00000] | 0.00560 / 96.80 / 5.8471, +0.00000 |
| s1c (W17 kernel) | s1c-decode-kernel 2e277290 (main 75726a0f) | 0.00490 / 96.75 / 7.0531, +0.00000 [+0.00000, +0.00000] | 0.00560 / 96.80 / 5.8471, +0.00000 |

Read: the release reproduces the 2026-11-17 reference row for row, so the gate is calibrated; W17's kernel gives the same
rows at both depths (bitwise in the scoring path, as its parity test says at 2,200 cells: `w17-parity.txt`). Rows are
`rows.tsv.xz` (`xz -d -k` to read).
