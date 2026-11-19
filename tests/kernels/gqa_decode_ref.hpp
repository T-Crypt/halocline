// tests/kernels/gqa_decode_ref.hpp - the decode launches the engine does not select, for the parity tests and the bench:
// the 16-row-tile launch (one block per (split, KV head, 16 rows), halo/kernels/gqa_attn.hpp: decode_mma_kernel) and the
// window kernel (one block per (split, KV head) holding every row of the window's KV head).  A parity test runs them
// beside the engine's launch on the same window and compares the rows; nothing in the engine selects them.
#pragma once

#include "halo/kernels/gqa_attn.hpp"

namespace halo::kernels::test {

/// gqa_decode_attn, forced onto the 16-row-tile kernel: the same arguments and the same output.
void gqa_decode_attn_tiled(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                           int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                           const int32_t* pos_dev = nullptr, KvInt8 q8 = {}, const float* gate_q = nullptr,
                           uint16_t* gated16 = nullptr, KvFar far_kv = {}, KvK8V4 k8v4 = {}, int64_t window = 0,
                           int64_t cap = 0, KvQ4 q4 = {});
/// gqa_decode_attn_segs, forced onto the 16-row-tile kernel: the same arguments and the same output.
void gqa_decode_attn_segs_tiled(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                                int64_t max_cells, float* out, void* stream, const float* gate_q = nullptr,
                                uint16_t* gated16 = nullptr);
/// gqa_decode_attn, forced onto the window kernel (one block per (split, KV head), every row of the window's KV head): the
/// same arguments and the same output.  Windows past 48 rows are the 16-row-tile launch, as they always were for it.
void gqa_decode_attn_window(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                            int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                            const int32_t* pos_dev = nullptr, KvInt8 q8 = {}, const float* gate_q = nullptr,
                            uint16_t* gated16 = nullptr, KvFar far_kv = {}, KvK8V4 k8v4 = {}, int64_t window = 0,
                            int64_t cap = 0, KvQ4 q4 = {});
/// gqa_decode_attn_segs, forced onto the window kernel: the same arguments and the same output.
void gqa_decode_attn_segs_window(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                                 int64_t max_cells, float* out, void* stream, const float* gate_q = nullptr,
                                 uint16_t* gated16 = nullptr);

}  // namespace halo::kernels::test
