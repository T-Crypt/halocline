// include/halo/kernels/kv_q4_layout.hpp - the q4_0 KV block, its quantizer and the Walsh-Hadamard rotation: what the
// sparse path stores 4-bit K/V in (src/models/qwen4exp/kernels/kv_q4.cu) and what the dense path's K8V4 cache and
// rotated 4-bit cache use, so one quantizer and one dequant rule serve both.
//
// A block is ggml's q4_0: an fp16 scale and 16 bytes of 4-bit codes for 32 values, value j in the low nibble of qs[j]
// and j + 16 in the high one. The scale is the block's signed largest-|value| over -8, so the codes span -8..7 and
// the dequantized value is (code - 8) * d. On its own that rounds a head's outlier channels to nothing; the rotation
// spreads them over all 256 dimensions first, which is what makes 4 bits usable.
#pragma once

#include "halo/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define HALO_KVQ4_HD __host__ __device__ __forceinline__
#else
#define HALO_KVQ4_HD inline
#endif

namespace halo::kernels {

inline constexpr int kKvQ4Block = 32;   // values one block holds (ggml's QK4_0)

#pragma pack(push, 1)
struct KvQ4Block {
    uint16_t d;                    // the block's scale, fp16 bits
    uint8_t qs[kKvQ4Block / 2];    // 32 codes: element j in the low nibble of qs[j], j + 16 in the high one
};
#pragma pack(pop)
static_assert(sizeof(KvQ4Block) == 18, "a q4_0 KV block is 18 bytes");

/// Bytes one KV head's row of blocks takes (head_dim / 32 blocks of 18).
inline uint64_t kv_q4_head_bytes(int head_dim) { return (uint64_t) (head_dim / kKvQ4Block) * sizeof(KvQ4Block); }

/// The block's delta from its signed largest-|value|: ggml's rule, that value over -8. A tie on |x| resolves to the
/// larger VALUE, so every lane of a warp (or a host loop in another order) picks the same scale. The codes are
/// rounded against this unrounded delta and the block stores it as fp16 (ggml's own q4_0 quantizer does the same).
HALO_KVQ4_HD float kv_q4_delta(float mval) { return mval / -8.0f; }

/// The block's stored scale bits: the delta as fp16.
HALO_KVQ4_HD uint16_t kv_q4_scale_bits(float mval) { return f16_from_f32(kv_q4_delta(mval)); }

/// A stored code's value: (code - 8) * d, d the STORED (fp16) scale as a float. The one dequant rule every reader of
/// a 4-bit KV row uses.
HALO_KVQ4_HD float kv_q4_dequant(int code, float d) { return (float) (code - 8) * d; }

/// A value's code against the block's delta, ggml's rule: trunc(x / d + 8.5), clamped to 0..15.
HALO_KVQ4_HD uint8_t kv_q4_code(float x, float d) {
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = (int) (x * id + 8.5f);
    return (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
}

/// One block from 32 contiguous values, on the host: the CPU reference and its tests. The scale is the same
/// lexicographic maximum of (|x|, x) that a warp reaches by shuffles, so the bytes are the kernel's.
inline KvQ4Block kv_q4_quantize_host(const float* x) {
    float amax = 0.0f, mval = 0.0f;
    for (int i = 0; i < kKvQ4Block; ++i) {
        const float v = x[i], a = std::fabs(v);
        if (a > amax || (a == amax && v > mval)) {
            amax = a;
            mval = v;
        }
    }
    KvQ4Block blk{};
    blk.d = kv_q4_scale_bits(mval);
    const float d = kv_q4_delta(mval);
    for (int i = 0; i < kKvQ4Block / 2; ++i)
        blk.qs[i] = (uint8_t) (kv_q4_code(x[i], d) | (kv_q4_code(x[i + 16], d) << 4));
    return blk;
}

/// The orthonormal 256-point Walsh-Hadamard transform (scale 1/16), rows of 256 contiguous floats: Hq . Hk = q . k,
/// and H is its own inverse, so the same call rotates V at the cache write and rotates the attention output back.
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) { fwht256_cuda(data, data, n_rows, stream); }

#if defined(__CUDACC__)
/// One 32-value group in one warp (lane = element): the stored scale bits, and `byte` is lane t's packed byte for
/// t < 16 (element t in the low nibble, element t + 16 in the high one). The largest |value| is reduced by shuffles
/// with the tie rule above, so all 32 lanes agree on it without shared memory. The statements are the ones the sparse
/// path's kernels shipped with, so the blocks they write are the same bytes.
__device__ __forceinline__ uint16_t kv_q4_group(float x, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    return f16_from_f32(d);
}
#endif

}  // namespace halo::kernels
