# The Ada (sm_89) port: what the 4090 lacks, and how the fallbacks are chosen

This page records how the engine runs on compute capability 8.9. The port started as basalt_ada's `fork_4090` branch and
was merged once with Basalt v0.4.1 (see `LIFTS.md`). All of it is development on one RTX 4090.

Numbers marked "development" were taken at a 450 W power limit, with the per-window log on unless a line says otherwise.
They are not the reference-machine numbers in the README, which were measured at 350 W on `SPECS.md`. Compare only runs
taken the same way.

## What sm_89 does not have (so what any compile error will be about)
Thread-block clusters and distributed shared memory, TMA / `cp.async.bulk` and mbarrier `expect_tx`/`try_wait`,
`griddepcontrol` (PDL), `fence.proxy.async`, `mul.bf16x2` (all sm_90+), and the block-scaled MMAs, `cvt e2m1` and the
direct `cvt.rn.bf16x2.e4m3x2` (sm_120a / sm_100+). It does have int8, bf16, f16 and plain e4m3 `mma.sync` (the e4m3 one
sums with too few bits for a W8A8 product: see the dense stack below), `cvt.rn.f16x2.e4m3x2`, `cp.async`, and 99 KB
opt-in shared memory per block.
nvcc here is whole-program (no rdc), so unreachable `__device__` helpers are dropped; only code reachable from a
compiled kernel must compile for sm_89.
Pattern used: wrap sm_90+ instructions in `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (or a local macro
`HC_SM90_ONLY` / `DM_SM90_ONLY` / `REC_NO_CLUSTER`), let the kernel trap below that, and make a **runtime** predicate pick
the fallback so the trap is never launched.

## How the fallbacks are chosen on the 4090
- GDN recurrence: `dec_gdnf_mode()` (src/kernels/cuda/gdn_window.cu since 0.4.0, a runtime function) returns 0 below sm_90, so
  `record_gdn.cpp` takes the unfused sequence. On 8.9 it runs as one persistent kernel, `k_mid_ada` (mid_chain_ada.cu),
  bitwise equal to the separate kernels, after a join on the L2 read-ahead. `--exp mid_ada=0` runs the separate kernels
  (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, z on native_mmvq), `--exp mid_ada=2` the kernel with z on
  dense_tc (other bits). Development A/B at 450 W: +2.2% decode (token-weighted) over `mid_ada=0`, faster on conv,
  struct and prose (ADA_MID_CHAIN_PLAN.md).
- Dense projections: the persistent `dense_tc` kernel (cp.async, int8 `mma.sync`) runs on sm_89; its bulk-copy variants
  (tile, one-tile, pair) stay off through `tc_bulk_ok()` (dense_tc.cu), and so does `qsa_qproj_half` (qsa_mid.cu).
- Lazy GDN commit: off below sm_90 (lifecycle.cpp): its read-ahead blocks wait beside the MoE grid and deadlock without
  PDL. The read-ahead itself (the next layer's projections and FFN read matrices into L2) runs alone instead, launched
  after each layer's MoE grid beside the combine (`gdn_l2_ahead`, `l2_ahead` in record_moe.cpp); `--exp l2_ahead=0`
  turns it off. Development A/Bs at 450 W: +0.7% and +1.5% decode (token-weighted), 21 of 24 paired runs faster;
  greedy output unchanged.
- Fused HC read and the sm_120 mid chain: already off unless the device major is 12 (hc_fused.cu `hc_fused_supported`).
- HC read: on 8.9 one persistent kernel, `k_hc_ada` (hc_read_ada.cu), bitwise equal to hc8_gr's three kernels, when no
  side branch is open, after a join on the L2 read-ahead. `--exp hc_ada=0` runs the three kernels. The router, the q8_1
  rows and the draft graphs stay outside it. Development A/B at 450 W, per-window log on, a download running: +3.17%
  decode (token-weighted) over `hc_ada=0`, faster on conv, struct and prose (HC_READ_ADA_PLAN.md).
- Route: without a peer, staged share or shared gate, `dec_moe_route` on 8.9 takes `dm_plan_ada_kernel` (block 0 plans, the
  others go straight to the q8_1 rows) and `dm_ring_ada_kernel` (one block publishes the header and x in 16-byte stores and
  rings; blocks for the warm-up's rows do not hold the doorbell back, so those rows may land after it, which changes no
  output). `--exp dm_route_ada=0` runs `dm_route_kernel`, which is unchanged. `dm_route_ada_parity`: 0 differing bytes in
  2560 cases against `dm_route_kernel`; greedy identical on conv, struct and prose. All numbers at 450 W, development only.
  Traces: chain 8-10 -> 5 us at T=1, 18-21 -> 7-9 us at T=8; ring 12-14 -> 8 us at T=1, 17-21 -> 12-14 us at T=8; prose's
  route busy time 7.6 -> 4.3 % of the cycle. Microbench, to the doorbell: 4.7 -> 3.3 us at T=1, 15.4 -> 6.2 us at T=8.
  A/B, 4 rounds x 2 repetitions with the video decoder idle throughout: conv -0.1 %, struct -0.7 %, prose +2.5 % (7 of 8
  faster), token-weighted +0.05 %. All three A/Bs pooled, 20 pairs each, paired median: conv +0.3 %, struct +0.5 %, prose
  +2.8 %, with prose's window time -3.3 %. So only prose gains beyond the spread; expert hit rates are unchanged. Other
  load on the machine moves these numbers: in an A/B during which a video played at some time, one conv run lost 21.5 %
  across every window width. Log `utilization.decoder` beside an A/B to see this.
- HC read and route together, against the build without them: `local_bench/engine_bench.py --tests community`, a fresh
  server from `halo_config.json`, 450 W, DDR5-4400, 4 rotated pairs, development only. conv / struct / prose /
  decode_tps 111.2 / 200.3 / 120.4 / 170.2 -> 118.7 / 212.3 / 127.3 / 180.1, +5.8 %, faster in every pair. decode_tps
  ranges 163.1 to 175.2 and 178.0 to 182.5. Same builds, plain halo-bench with the log on, 14 pairs: +5.53 %; the hc
  read alone +3.53 % and the route on top +1.40 %, 6 pairs each. Text differs between runs on conv and prose (adaptive
  swaps), so struct, greedy, is the cleanest figure.
- Those builds' output: the quality gate's rows (100 chunks, T 3) are byte-identical; greedy, and with `--adapt-swaps 0`
  sampled conv and prose, are identical.
- Measuring: plain halo-bench asks for the per-window log unless `--no-windows`, about 2 % slower, so compare its tok/s
  only with its own. It keeps the last of a repeated engine flag, so two experiment gates go in one `--exp a=0,b=0`.
- Comparing the quality gate across builds: its rows follow the GPU slots, and a bigger binary leaves less VRAM (6491 slots
  against 6515 changed nearly every row, KLD 0.0620 against 0.0619). Fix `--expert-cache` when comparing; with it equal the
  rows are byte-identical.
- DeepSeek-V4.1 on 8.9: its code uses no bulk copies, griddepcontrol or FP4 tensor cores; DSpark decodes MXFP4 in
  software. Since upstream 0.3.1+13 one kernel uses a cluster: the index key (`index_key_kernel`, attn_index.cu, eight
  blocks a token). Below 9.0 it compiles without its cluster and traps, and `index_keys()` runs `index_key_block_kernel`
  instead: a block a token, the same row sums (`index_key_row`) and the same finish (`index_key_finish`), so the keys keep
  the cluster form's bits (`block_sum`'s extra warps add zeros). sm_120a's SASS of attn_index.cu is unchanged apart from the
  added kernel. Its `ds41_*` parity tests (`-DBUILD_TESTING=ON`) pass on a 4090 at 450 W, the peer test only with one card.
  No pack has been run, so end-to-end output, quality and speed are unmeasured.
- Frozen hashes: `ds41_router_parity`, `ds41_attn_index_parity` (gemv_bf16, compressor rows and keys) and
  `ds41_dspark_parity` hash a kernel's output against the bits of the kernel it replaced, frozen upstream on Blackwell.
  On the 4090 the replaced kernels (fork_4090's sources before 0.3.1+13) give other bits, and their replacements give
  exactly those: router f4ec7c76f483174c, gemv 258e734e874cae57, compressor d2ec8ec03dfea810, DSpark MoE c7b1acee87ad0e77.
  `frozen_for_device` (kernels/test_util.hpp) uses these below 9.0, upstream's values from 9.0 up.
- Since 0.3.2+80 DSpark's MoE rows take q8_1 activations and dp4a (upstream 30ad59a6), which changes their bits by design;
  upstream froze the new kernel's own bits (3be9ade2c87c72d7). dspark_kernels.cu has no architecture-dependent code, and
  every other check in `ds41_dspark_parity` passes on the 4090 (against the double reference, the fp32 hidden rows, live
  rows bitwise), so the 4090's bits of the same kernel are its frozen value below 9.0: 9c13389dbf259e5f. Unlike the values
  above, this one was not checked against a replaced kernel on the 4090: it pins the dp4a kernel's bits for later changes.
- Draft reads: on 8.9 the draft graphs' hyper-connection reads, their router and q8_1 rows run as one kernel,
  `hc_fused_ada` (hc_fused_ada.cu), where `hc_fused` does not run. Verify stays on `k_hc_ada`, and `--exp hc_fused_ada=0`
  keeps the separate kernels. A one-row read now takes fused_gr_read_multi's bits instead of gr_read's, which only the
  drafts see: greedy and sampled output with swaps off, and the gate's rows at equal slots, are identical.
  `hc_fused_ada_parity`: 0 differing cases at T 1..8, 0 of 208 stress reads, 0 stuck with 100 SMs held. Microbench at
  450 W, T = 1, hot / cold: read 24.4 / 32.7 -> 12.5 / 26.3 us; with the router 38.1 / 44.2 -> 17.0 / 33.4 us. The same
  binary on against off at 450 W, development only: struct +0.15 % (10 pairs), conv +0.7 % and prose +1.8 % pooled over 8
  seeds; on identical text conv +0.5 %, prose about +1.3 %. The traces suggested about 1.4 % on struct; the GDN commit
  running beside the draft may be why it does not show. Struct runs near 199 tok/s were seen with it, but in the same-binary
  A/B the worst dip (203) was with it off.
- Measuring conv and prose: their text path moves them more than most changes do, so use several seeds. Seed 7 alone
  showed this kernel at +4.4 % on conv.
- 2026-11-17 (Halocline 2c1): the MoE experts' FP4 prompt path (`src/prefill/moe_fp4.cu`, library `prefill_fp4`,
  `expert_fp4.cpp`, `prefill_fp4_test`) is removed: it was `120a-real` only and never ran on 8.9. `fp4_dense_tma_test`
  stays (it covers the dense product's 8.9 kernel), without its comparison against the removed path.
  NVFP4 experts always take MMQ; the notes below on `fp4_tensor_cores()` describe the code before that.
- NVFP4 prefill: `fp4_tensor_cores()` (layout.cpp) is true only for major >= 12. `fp4_pack()`, `fp4_every_layer()` and
  `fp4_l` (experts.hpp) depend on it, so NVFP4 experts take the MMQ path (src/prefill/mmq, mmq_nvfp4) and the MMQ
  buffers are sized for every chunk length. If a buffer overflow or wrong-size error shows up in prefill, look at
  `moe_bufs` / `group_rows_cap` in layout.cpp first.
- Sampler / qsa_select cluster kernels check the device major. PDL launches (`launch_pdl`, `launch_dependent`) check
  `pdl_kernel_ok` and launch plain below 9.0: a programmatic edge without griddepcontrol only adds latency, 25 to 42 us
  once a fork/join has moved the chain to another of the graph's internal streams.
- The VRAM TMA expert kernel (`dec_moe_vram` kind 1) is never selected by default.

## The dense stack (Qwen3.8-27B) on 8.9 (upstream 0.4.0)
- No 27B pack has been run (none downloaded; `qwen35_checkpoint_round_trip` skips without one). The pack takes 16.0 GiB
  on the card with its drafter, so it would fit a 4090. Every kernel the stack calls has an 8.9 path, checked by
  upstream's kernel tests on the 4090.
- Upstream builds `window_gemm`, `prefill_fp8`, the dense FP4 product and two tests (`narrow_bf16_parity`, `silu_parity`)
  for `120a-real` only. Halocline builds them for `89-real` (`HALO_CUDA_ARCHS_A` in CMakeLists.txt). The dense FP4
  product is its own library, `prefill_fp4_dense`.
- Decode windows: `window_gemm` (cp.async, bf16 mma) runs as on 12.0. On 8.9 its NVFP4 codes decode by a byte permute
  from the eight magnitudes' patterns, and its scale multiply is `fma.rn.bf16x2` with a -0 addend (narrow_bf16.cuh): the
  same bits as 12.0a's conversions and `mul.rn.bf16x2` (`narrow_bf16_parity`: every input against the host values).
  `dec_gdn_rec` runs as `dec_gdn_rec_split` + `dec_gdn_rec_norm` there (the same bits, `state_out` included; the raw
  output in `a.o`, or else `a.y`); batch windows use `dec_gdn_rec_seg_split` on every card; `dec_gdn_rec_seg` refuses
  below 9.0.
- Prompt chunks. `gdn_gating_proj`: a flat form, a block of 8 warps per 16 tokens, warp q the cluster form's rank q (a
  copy of its loop, so the cluster kernel stays upstream's source), the partials added in rank order. FP8 W8A8
  (`fp8::gemm`): 8.9's FP8 mma keeps too few bits in its sums (1.2e-3 of sum |w a| at K 6144, 1.5e-4 with each k32
  step's sum moved into fp32, against the test's 1e-5), so the codes decode to f16 (exact) and the product runs on f16
  mma with fp32 accumulation: 1.6e-6, 134-161 TFLOP/s at 450 W. On the FP8 tensor cores the same kernel reaches 220-268
  TFLOP/s (1.65-1.85x) at 1.5e-4 even with the sums in fp32; open as item 6 of ADA_NOT_IMPLEMENTED.md (main checkout,
  untracked), with the measurements. Only packs with F8_E4M3 projections (NVIDIA's 27B checkpoint) call it. FP4 W4A4 (`fp4_dense`, 12.0's FP4 tensor cores): each code
  times its UE4M3 scale decoded to f16 (exact), f16 mma, the TMA kernel's epilogue arithmetic; the quantizer rounds to
  E2M1 in software (nvfp4_planes.cu's rounding): max err 6.4e-8 of sum |w x|, rows bitwise at any chunk length, 97-128
  TFLOP/s at 450 W. The products are exact; the sums run in another order than 12.0's block-scaled mma, so a 4090's
  prompt rows are not a 5090's bits.
- sm_120 code: the SASS of every upstream kernel in the touched files (dense_ops, gdn_window, window_gemm*, fp8_gemm,
  fp4_dense) is identical to upstream 0.4.0's; the 8.9 kernels are trap stubs there.
- `dense_ops_parity` fails only its old baseline, `bf16_rows_dot` + `gdn_gates` at T = 2048 (3.17e-4 against 3e-4, beta
  3.37e-5 against 3e-5): its tf32 mma accumulates below fp32 (the test's own comment), and the engine no longer calls it.
  `bf16_rows_dot` itself passes. The thresholds were left as they are.

