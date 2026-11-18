// tests/kernels/gqa_decode_ref.hpp - the decode window's 16-row-tile launch (one block per (split, KV head, 16 rows),
// halo/kernels/gqa_attn.hpp: decode_mma_kernel), which the window kernel replaced for windows up to 48 rows.  A parity
// test runs both launches on the same window and compares them bit for bit; nothing in the engine selects this one.
#pragma once

#include "halo/kernels/gqa_attn.hpp"

namespace halo::kernels::test {

/// gqa_decode_attn, forced onto the 16-row-tile kernel: the same arguments and the same output.
void gqa_decode_attn_tiled(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                           int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                           const int32_t* pos_dev = nullptr, KvInt8 q8 = {}, const float* gate_q = nullptr,
                           uint16_t* gated16 = nullptr, KvFar far_kv = {}, KvK8V4 k8v4 = {}, int64_t window = 0,
                           int64_t cap = 0);
/// gqa_decode_attn_segs, forced onto the 16-row-tile kernel: the same arguments and the same output.
void gqa_decode_attn_segs_tiled(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                                int64_t max_cells, float* out, void* stream, const float* gate_q = nullptr,
                                uint16_t* gated16 = nullptr);

}  // namespace halo::kernels::test
