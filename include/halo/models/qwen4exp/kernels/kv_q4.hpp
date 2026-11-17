// include/halo/models/qwen4exp/kernels/kv_q4.hpp - Q4_0 KV storage with Walsh-Hadamard rotation for the QSA layers (`--kv q4_0`).
//
// K and V are rotated by the orthonormal 256-point Hadamard matrix H before they are
// quantized to ggml's q4_0 (32 values per block: an fp16 scale + 16 bytes of 4-bit codes, 144 B per head per cell,
// 576 B per cell for K and V of both heads, vs 1,056 B in INT8). The rotation spreads a head's outlier channels
// over all 256 dimensions, which is what makes 4 bits usable. The query is rotated the same way, so
// <Hq, Hk> = <q, k> and the scores are unchanged, and the attention output (a mix of rotated values) is rotated
// back with H (self-inverse). What is lossy is the 4-bit rounding itself, so check quality before using it.
//
// The block, its quantizer and the rotation live in halo/kernels/kv_q4_layout.hpp, shared with the dense path's 4-bit
// KV formats (`--kv k8v4`, and rotated 4-bit K and V): one quantizer serves both, so a block written here and a block
// written there are the same bytes.
//
// With KV streaming (--kv-resident) the codes live in the host copy and the resident slots like INT8's (KvHostPools
// k_q4/v_q4, the same block granule), so both options combine.
#pragma once

#include "halo/kernels/kv_q4_layout.hpp"
#include "halo/models/qwen4exp/kernels/qsa.hpp"

#include <cstdint>

namespace halo::kernels {

inline constexpr int QK4_0 = kKvQ4Block;
using block_q4_0 = KvQ4Block;

/// Bytes per cell and KV head: 8 blocks of 32 for head_dim 256 -> 144.
inline uint64_t kv_q4_bytes_per_head(int head_dim) { return kv_q4_head_bytes(head_dim); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_q4_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q4_bytes_per_head((int) s.head_dim) * 2;
}

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                  const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream);

}  // namespace halo::kernels
