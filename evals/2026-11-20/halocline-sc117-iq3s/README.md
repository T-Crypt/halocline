# SC117 GSQ-RCO abliterated IQ3_S on Halocline, against UD-IQ4_XS, at 350 and 450 W (2026-11-20)

Branch `abil-halo`. Source: SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF, revision `90a3350c`, `IQ3_S/` (both
shards match the Hugging Face sha256: shard 1 `40f369ec...`, shard 2 `316b46f3...`). Question: how fast does
Halocline run the quant Adamant headlines (GSQ-RCO IQ3_S, 190 / 372 / 169 tok/s conv / struct / prose at 450 W), and how
much of a gap is quant and how much is power?

## Pack

`convert.py --gguf <shard 1> --sections-from <served UD pack>` (drafter and vision sections copied from the UD pack;
SC117's MTP head is the stock base model's): 87,889,249,268 bytes in 6.5 min, every one of the tier's 1,223 tensors
accepted (`convert.txt`, `convert-dry.txt`). The first dry run failed on a converter bug: a dry run records empty
hashes, so the `--sections-from` copy check failed every time. Fixed in 71199e25 (on `main` as 6ba99fcf).

## Bench

`halocline-bench --community` (preset community-1: conv, struct and prose decode fixtures, a 16K and a 64K prefill,
each in a cold engine, int8 KV, no other flags), built from the `abil-halo` checkout at 71199e25 (the preset needs a
git build; release trees are archives). RTX 4090, i9-14900K, 62 GiB RAM. `queue.sh` and `bench.sh` ran the four under
one Berth lease; `gpu_tune.py` raised only the power limit for C and D (clock lock 2,600 MHz and +150 offset kept), and
`gpu-tune.service` restored 350 W at the end.

| Run | Pack | Power | conv | struct | prose | decode (weighted) | 16K prefill | 64K prefill | conv accept |
|---|---|---|---|---|---|---|---|---|---|
| A | UD-IQ4_XS | 350 W | 135.4 | 266.7 | 138.6 | 218.2 | 5,709 | 4,018 (cold cache) | 71.5% |
| B | SC117 IQ3_S | 350 W | **171.7** | **316.0** | **154.1** | **256.1** | 5,610 | 6,673 | 76.3% |
| C | SC117 IQ3_S | 450 W | 160.2 | 316.4 | 151.4 | 255.9 | 5,637 | 6,737 | 68.1% |
| D | UD-IQ4_XS | 450 W | 141.9 | 265.1 | 135.0 | 218.8 | 5,757 | 6,572 | 74.0% |

Decode in tok/s, prefill in engine tok/s. Raw rows: `*.bench.txt` and the branch's `docs/community/benchmarks.json`.

## Reading

- **The quant:** SC117 IQ3_S decodes 11-27% faster than UD-IQ4_XS at the same power (B against A). The engine loads
  51.4 GiB against 60.9, so more experts stay resident.
- **Power does nothing on this card.** At a 450 W limit the card peaked at 391 W and 60-61 C with no slowdown sample
  (`thermal-C-*.csv`, `thermal-D-*.csv`, 1 s samples): the 2,600 MHz clock lock holds it at 2,595 MHz before the power
  limit binds. C matches B and D matches A within noise. Conv moves with draft acceptance (68-76%), not with power.
- **Against Adamant:** at 350 W, B reaches 90% / 85% / 91% of Adamant's 450 W conv / struct / prose. Power does not
  explain the rest. Unmeasured candidates: Adamant's clocks (their docs give a 450 W limit and no clock settings)
  and engine or config differences. Only an Adamant run on this machine would separate those.
- **64K prefill is not comparable across these runs.** A ran right after the converter had read 84 GB of SC117 through
  the page cache, evicting the UD pack; D, on the same pack, reads 6,572. The n-gram table is read from the pack during
  prefill (see 2026-11-19 `halocline-sparse-abba`), so this number tracks the page cache. Decode does not.

## Not done here

Quality: no KLD against BF16, no smoke, no vision check on the SC117 pack yet. The abliteration changes outputs on
purpose, so KLD against the base model measures the edit plus the quant and is information, not a gate. No llama-swap
entry yet: `examples/qfn-sc117-iq3s.json` on the branch is the local config.

`failed-1154/`: the first bench attempt, which stopped at once (release trees cannot run `--community`) and whose
thermal watchdog tripped on nvidia-smi's "Not Active" text. Both fixed before the runs above.
