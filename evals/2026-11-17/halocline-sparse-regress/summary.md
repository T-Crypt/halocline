# Halocline sparse non-regression gate, 2026-11-17T19:27:13-07:00

REF `~/halocline/releases/d314a21a` (bin/basalt) vs NEW `~/halocline/releases/b22ecbb2` (bin/halocline). Pack `/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt`.
Quality base `flashnext-bf16-base.kld`; int8 KV; drafter `--spec 8` except gate 1 (`--spec 0`).
Power limit 350 W. Wall time 4419 s.

| # | gate | verdict | numbers |
|---|---|---|---|
| 1 | greedy identity (spec 0) | PASS | 900/900 tokens, GREEDY IDENTICAL |
| 2 | KLD + speed (int8 KV, spec 8) | FAIL | KLD REF 0.08195 / NEW 0.08195 (paired n/a); worst speed delta 31.8% (limit 3%) |
| 3 | sampling distribution (temp 1.0) | PASS | top1 pass, top10 pass, logprob pass, first_tv pass |
| 4 | tool-call breadth (temp 0) | PASS | REF 24/24, NEW 24/24, regressed none |

## 1. Greedy identity detail

```
ref: 900 tokens, 3 requests
new: 900 tokens, 3 requests
GREEDY IDENTICAL
```

## 2. KLD and speed detail

The published int8 numbers for comparison: KLD 0.08195, top-1 90.45%, conv 131.8 / struct 264.6 / prose
133.7 tok/s, prefill 6,053 / 6,019 t/s (halocline-kv-kld/README.md:25,37).

| metric | REF1 | REF2 | REF mean | NEW1 | NEW2 | NEW mean | NEW-REF % | published int8 |
|---|---|---|---|---|---|---|---|---|
| conv tok/s | 135.2 | n/a | 135.2 | n/a | 134.9 | 134.9 | -0.2 | 131.8 |
| struct tok/s | 266.2 | n/a | 266.2 | n/a | 264.5 | 264.5 | -0.6 | 264.6 |
| prose tok/s | 137.0 | n/a | 137.0 | n/a | 136.8 | 136.8 | -0.1 | 133.7 |
| prefill 32K t/s | 5785.1 | n/a | 5785.1 | n/a | 5826.5 | 5826.5 | +0.7 | 6053.0 |
| prefill 131K t/s | 1841.5 | n/a | 1841.5 | n/a | 2427.7 | 2427.7 | +31.8 | 6019.0 |

| run | chunks | rows | PPL | mean KLD | top-1 % |
|---|---|---|---|---|---|
| ref1 | 100 | 25500 | 3.2323 | 0.08195 | 90.45 |
| ref2 | n/a | n/a | n/a | n/a | n/a |
| new1 | n/a | n/a | n/a | n/a | n/a |
| new2 | 100 | 25500 | 3.2323 | 0.08195 | 90.45 |

Paired KLD (kv_paired.py): the 95% interval of NEW minus REF, resampling chunks. Gate 2 needs it to hold
0 on both pairs, and every speed metric inside 3% on the ABBA means.

`paired-ref1-new1.md`:

new1: no rows.tsv
| KV | rows | PPL | base PPL | mean KLD | top-1 % | paired dKLD vs ref1 [95% CI] | top-1 points |
|---|---|---|---|---|---|---|---|
| ref1 | 25500 | 3.2323 | 3.2733 | 0.08195 | 90.45 | (reference) |  |

`paired-ref2-new2.md`:

ref2: no rows.tsv
| KV | rows | PPL | base PPL | mean KLD | top-1 % | paired dKLD vs ref2 [95% CI] | top-1 points |
|---|---|---|---|---|---|---|---|

`paired-ref1-ref2.md`:

ref2: no rows.tsv
| KV | rows | PPL | base PPL | mean KLD | top-1 % | paired dKLD vs ref1 [95% CI] | top-1 points |
|---|---|---|---|---|---|---|---|
| ref1 | 25500 | 3.2323 | 3.2733 | 0.08195 | 90.45 | (reference) |  |

`paired-new1-new2.md`:

new1: no rows.tsv
| KV | rows | PPL | base PPL | mean KLD | top-1 % | paired dKLD vs new1 [95% CI] | top-1 points |
|---|---|---|---|---|---|---|---|

## 3. Sampling detail

```
metric             A   cand - A     reference B - A, 95%  verdict   (24 prompts)
top1          0.8725    +0.0000 [-0.0048, +0.0024]  pass
top10         0.9987    +0.0000 [-0.0010, +0.0009]  pass
logprob      -0.3585    +0.0000 [-0.0093, +0.0085]  pass
first_tv                 0.0000 [0.0273, 0.0781]  pass
PASS
```

## 4. Tool-call detail

REF:

```
call  12/12
none  12/12
all   24/24
```

NEW against REF:

```
call  12/12
none  12/12
all   24/24
agreement with the reference: 24/24
```

## Acceptance

* Gate 1: `greedy_identity.py` exit 0, `GREEDY IDENTICAL`, 900 of 900 tokens on both sides.
* Gate 2: the paired 95% interval of NEW minus REF holds 0 on both ABBA pairs, and every speed metric is
  within 3% on the ABBA means. A REF that misses the published numbers is shown in the table, not scored.
* Gate 3: `sampling_dist.py compare` exits 0 and its last line is `PASS` (top1, top10, logprob and first_tv
  inside the REF build's own seed-change interval).
* Gate 4: no case REF answers right is wrong on NEW (the per-case JSONL comparison); accuracy is reported.
* Gate 1 and 2 configurations are the published ones verbatim; gates 3 and 4 pin placement
  (`--expert-cache 7000 --adapt-swaps 0 --max-context 8192`) so adaptation cannot move experts between runs.

