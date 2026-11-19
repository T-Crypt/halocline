# Sparse prefill at depth: the RAM reserve and the pack disk (2026-11-19)

Question from the vision change (`../halocline-vision`): the served sparse entry moved to the vision pack and
`--ram-reserve-mib 4096`, and its long smoke read slower prompts than the night before (36K 4,468 vs 6,311 t/s, 137K
4,867 vs 5,832). Is that the reserve, the new file's page cache, or something else?

Release 5aec8e3a, served sparse flags (`--spec 8 --prefill auto --max-context 262144 --kv int8`) on :8096,
`lsw.py smoke --long`, under `berth train`. Power limit 350 W. `run.sh` (arms), `nvme-step.sh` (the copy and the second
series). `read_during_smoke_mib` is the server tree's `/proc/*/io` read_bytes from "weights loaded" to the end of the
smoke (bytes that came from the disk, not the page cache).

## contended/ (discarded)

The first series ran while DeepSeek workers were running the server test suite on all 32 threads; the sparse path's
experts run on the CPU, so both arms read ~2,000 t/s at 137K. Stopped after arm 3; the workers were pinned to E-cores
28-31 (`taskset`) for everything below.

## Results

| series | arm | pack disk | reserve | vision | 36K prompt | 137K prompt | 137K TTFT | disk read in smoke |
|---|---|---|---|---|---|---|---|---|
| 1 | r2048 | SATA (`/data`) | 2048 | off | **5,967** | 2,678 | 45.9 s | 20.3 GiB |
| 1 | r4096 | SATA | 4096 | cpu | 4,055 | 3,262 | 42.9 s | 14.0 GiB |
| 2 | nvme | NVMe (`/mnt/models`) | 4096 | cpu | 4,631 | **4,417** | 27.2 s | 29.4 GiB |
| 2 | r4096 | SATA | 4096 | cpu | 4,467 | 1,857 | 63.7 s | 28.6 GiB |
| 2 | r4096 | SATA, page cache warm from the arm before | 4096 | cpu | 4,463 | 4,908 | 24.6 s | 4.2 GiB |
| 2 | nvme | NVMe | 4096 | cpu | 4,660 | **4,106** | 29.2 s | 31.1 GiB |

Decode is 116-125 t/s in every arm (the experts' placement and the disk do not enter decode once the prompt is in).

## Read

1. **The pack disk sets the 137K prefill when the page cache is cold.** The same file (byte-identical, `cmp`) reads
   1,857 to 3,262 t/s from the SATA SSD and 4,106 to 4,417 from the NVMe. With the file warm in the page cache, SATA
   reaches 4,908 and reads 4.2 GiB instead of ~29. The SATA disk (WD WDS200T2B0B) also holds `/`, so any other disk
   work on the host lands on the same device; the day-to-day spread at 137K (1,918 to 5,832 across the last runs) is
   this. `SPECS.md` says an NVMe pack disk only speeds up the cold start; that is wrong for prefill at depth.
2. **The 4096 reserve costs about a quarter of the 36K prompt rate.** 5,967 at 2048 against 4,055 to 4,660 at 4096 on
   either disk; the expert RAM plan shrinks from 47.1 GB to 43.3-44.8 GB. One 2048 arm only (with vision off), so the
   size is indicative. The reserve exists because the vision sidecar's ~1.1 GiB is not in the engine's RAM plan;
   making the sidecar's projector file-backed (worker W31, 2026-11-19) is the way back to 2048.
3. Not separated here: how much of the ~29 GiB read in a cold smoke is the n-gram table and how much is other file
   reads. The table reader (`include/halo/ngram/ple_reader.hpp`) reads 4 KiB pages; a 137K prompt needs 16 rows a
   token.

## Copy

`/mnt/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp-vision.halo`: `dd iflag=direct oflag=direct` in 539 s,
`cmp` identical. The served entry still points at `/data`.
