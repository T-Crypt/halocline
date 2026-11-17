// src/prefill/fp8_gemm.cu - see include/halo/prefill/fp8_gemm.hpp.
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
// Upstream's product (Infernix's fp8_a8_tma_mma_kernel: TMA tiles through an mbarrier ring, the block-scaled e4m3 mma)
// needs compute capability 9.0 and was removed.  What runs here:
// Compute capability 8.9 has neither TMA, mbarrier transaction counts nor the block-scaled form, and its plain e4m3 mma keeps
// fewer bits than fp32 in its sums, even inside one k32 step (1.5e-4 of sum |w a| at K 6144 against 1e-5 here, Hopper's
// known FP8 accumulation).  There gemm_ada_kernel decodes the codes to f16 (every E4M3 value is exact in f16) and runs the
// product on mma m16n8k16 with fp32 accumulation: the block tile, the eight warps over it, the K loop in order, the same
// epilogue arithmetic; stages of 64 K filled by cp.async from every thread, and the BF16 outputs go straight from the
// accumulators.  Its products are exact; its sums run in another order than the block-scaled mma's.
#include "halo/prefill/fp8_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>

#include "halo/log.hpp"


namespace halo::prefill::fp8 {
namespace {

constexpr int BT = 128, BR = 256;                       // tokens, weight rows per block
constexpr int WT = 2, WR = 4;                           // consumer warps over tokens, rows
constexpr int NCONS = WT * WR * 32;                     // threads
constexpr int WTOK = BT / WT, WROW = BR / WR;           // a warp's 64 tokens x 64 rows
constexpr int MT = WTOK / 16, MR = WROW / 8;            // its mma tiles: 4 x 8
constexpr int MAX_SEG = 3;

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        halo::log::error("halo", "prefill fp8: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Seg {
    const float* scale;
    void* dst;
    int64_t ld;
    int rows;
    int tile0;   // first row tile of this segment in the grid
};

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
// 16 bytes to shared memory, the first `n` of them from src and zeros after
__device__ __forceinline__ void cp16z(void* dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(su(dst)), "l"(src), "r"(n) : "memory");
}

// ---------------------------------------------------------------------------------------- compute capability 8.9
constexpr int A_BK = 64, A_ST = 4;                       // K per stage (bytes per row), stages
constexpr int A_ACT = BT * A_BK, A_STAGE = A_ACT + BR * A_BK;
constexpr int A_SMEM = A_ST * A_STAGE;
static_assert(A_SMEM <= 99 * 1024 && (BT * 4) % NCONS == 0 && (BR * 4) % NCONS == 0, "the Ada stages");

struct AdaArgs {
    const uint8_t* act;          // the activation codes [ntok][K]
    const uint8_t* w[MAX_SEG];   // each segment's weight codes [rows][K]
    Seg seg[MAX_SEG];
    const float* a_scale;
    int64_t K;
    int nseg, ntok, tok_tiles, k_tiles, epi;
};

// two E4M3 codes (the low 16 bits of x, the low byte the lower half) as f16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_f16x2(unsigned x) {
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f;
}
// m16n8k16 f16 x f16 -> f32
__device__ __forceinline__ void mma_f16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// byte offset of (row, logical byte) in a stage tile of 64-byte rows: the 16-byte pieces swizzled by (row / 2) % 4, so
// the eight rows of an ldmatrix matrix fall in eight bank groups
__device__ __forceinline__ int swa(int row, int byte) {
    return row * A_BK + ((((byte >> 4) ^ (row >> 1)) & 3) << 4) + (byte & 15);
}

template <bool BF16>
__global__ void __launch_bounds__(NCONS, 1) gemm_ada_kernel(const __grid_constant__ AdaArgs p) {
    extern __shared__ __align__(128) unsigned char raw[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wtk = warp / WR, wrw = warp % WR;
    const int row_tile = blockIdx.x / p.tok_tiles, tok_tile = blockIdx.x - row_tile * p.tok_tiles;
    int si = 0;
    for (int i = 1; i < p.nseg; ++i)
        if (row_tile >= p.seg[i].tile0) si = i;
    const int row0 = (row_tile - p.seg[si].tile0) * BR, tok0 = tok_tile * BT;
    const Seg sg = p.seg[si];
    const uint8_t* wsrc = p.w[si];
    // stage st <- K tile kt: the activation rows' four 16-byte pieces, then the weight rows' (zeros past the tensors)
    const auto load = [&](int kt, int st) {
        unsigned char* as = raw + st * A_STAGE;
        unsigned char* ws = as + A_ACT;
        const int64_t k0 = (int64_t) kt * A_BK;
#pragma unroll
        for (int i = 0; i < BT * 4 / NCONS; ++i) {
            const int idx = tid + i * NCONS, r = idx >> 2, pc = idx & 3, t = tok0 + r;
            const bool ok = t < p.ntok;
            cp16z(as + swa(r, pc * 16), p.act + (int64_t) (ok ? t : 0) * p.K + k0 + pc * 16, ok ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < BR * 4 / NCONS; ++i) {
            const int idx = tid + i * NCONS, r = idx >> 2, pc = idx & 3, wr = row0 + r;
            const bool ok = wr < sg.rows;
            cp16z(ws + swa(r, pc * 16), wsrc + (int64_t) (ok ? wr : 0) * p.K + k0 + pc * 16, ok ? 16 : 0);
        }
    };
    float acc[MT][MR][4] = {};
    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int st = 0; st < A_ST - 1; ++st) {
        if (st < p.k_tiles) load(st, st);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll 1
    for (int kt = 0; kt < p.k_tiles; ++kt) {
        asm volatile("cp.async.wait_group %0;" ::"n"(A_ST - 2) : "memory");
        __syncthreads();   // stage kt is in, and every warp has left stage kt - 1, which the next load refills
        if (kt + A_ST - 1 < p.k_tiles) load(kt + A_ST - 1, (kt + A_ST - 1) % A_ST);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const unsigned char* as = raw + (kt % A_ST) * A_STAGE;
        const unsigned char* ws = as + A_ACT;
        // four k16 steps: thread (g, c) takes values 16c .. 16c + 15 of the stage for its rows, step s's k-slots {2c, 2c+1}
        // values 16c + 4s, +1 and {2c+8, 2c+9} values 16c + 4s + 2, +3 (one 4-byte read a row a step), the same for both
#pragma unroll
        for (int s = 0; s < A_BK / 16; ++s) {
            unsigned af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r = wtk * WTOK + mt * 16 + g;
                const unsigned w0 = *reinterpret_cast<const unsigned*>(as + swa(r, 16 * c + 4 * s));
                const unsigned w1 = *reinterpret_cast<const unsigned*>(as + swa(r + 8, 16 * c + 4 * s));
                af[mt][0] = e4m3x2_f16x2(w0);
                af[mt][1] = e4m3x2_f16x2(w1);
                af[mt][2] = e4m3x2_f16x2(w0 >> 16);
                af[mt][3] = e4m3x2_f16x2(w1 >> 16);
            }
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {   // each weight fragment decoded right before its products
                const unsigned w = *reinterpret_cast<const unsigned*>(ws + swa(wrw * WROW + mr * 8 + g, 16 * c + 4 * s));
                const unsigned b0 = e4m3x2_f16x2(w), b1 = e4m3x2_f16x2(w >> 16);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) mma_f16(acc[mt][mr], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b0, b1);
            }
        }
    }
    const float ws = sg.scale ? *sg.scale : 1.0f;
    const bool add = p.epi == (int) Epilogue::Add || p.epi == (int) Epilogue::AddBf16;
    // gemm_kernel's epilogue arithmetic, a token row at a time (with Add, its residual pairs loaded before any is written)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tk = tok0 + wtk * WTOK + mt * 16 + (lane >> 2) + h * 8;
            if (tk >= p.ntok) continue;
            const float as = __ldg(p.a_scale + tk);
            float2 res[MR];
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + wrw * WROW + mr * 8 + 2 * (lane & 3);
                res[mr] = make_float2(0.0f, 0.0f);
                if (add && r < sg.rows) {
                    if constexpr (BF16)
                        res[mr] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                            static_cast<const __nv_bfloat16*>(sg.dst) + (int64_t) tk * sg.ld + r));
                    else
                        res[mr] = __ldcg(reinterpret_cast<const float2*>(static_cast<const float*>(sg.dst) + (int64_t) tk * sg.ld + r));
                }
            }
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + wrw * WROW + mr * 8 + 2 * (lane & 3);
                if (r >= sg.rows) continue;
                // rounded products, then the add: no contraction into an fma, so the bits do not depend on the instance
                float2 v = make_float2(__fmul_rn(__fmul_rn(acc[mt][mr][2 * h], as), ws),
                                       __fmul_rn(__fmul_rn(acc[mt][mr][2 * h + 1], as), ws));
                if (add) v = make_float2(__fadd_rn(res[mr].x, v.x), __fadd_rn(res[mr].y, v.y));
                if constexpr (BF16)
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(sg.dst) + (int64_t) tk * sg.ld + r) =
                        __floats2bfloat162_rn(v.x, v.y);
                else
                    *reinterpret_cast<float2*>(static_cast<float*>(sg.dst) + (int64_t) tk * sg.ld + r) = v;
            }
        }
}

size_t scale_off(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~(size_t) 15; }

// One block per row: amax over the row, then E4M3 codes of x * 448 / amax (saturating, round to nearest even).
// NORM: the row is RMSNorm'd first (x * rsqrt(mean(x^2) + eps) * nw), the normed row optionally written to xn.
// In: float or __nv_bfloat16 rows (xn, when written, in the same type).
__device__ __forceinline__ float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 load4(const __nv_bfloat16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void store4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void store4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
}
template <bool NORM, class In = float>
__global__ void __launch_bounds__(256) quantize_kernel(const In* __restrict__ x, int64_t cols, int64_t ld,
                                                      uint8_t* __restrict__ codes, float* __restrict__ scale,
                                                      const float* __restrict__ nw = nullptr, float eps = 0.0f,
                                                      In* __restrict__ xn = nullptr) {
    __shared__ float red[8];
    __shared__ float inv_s, inv_rms_s;
    const In* xr = x + (int64_t) blockIdx.x * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
            const float4 v = load4(xr + i);
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < 8; ++i) t += red[i];
            inv_rms_s = rsqrtf(t / (float) cols + eps);
        }
        __syncthreads();
        inv_rms = inv_rms_s;
    }
    auto val4 = [&](int64_t i) -> float4 {
        float4 v = load4(xr + i);
        if (NORM) {
            const float4 w = *reinterpret_cast<const float4*>(nw + i);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float m = 0.0f;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = val4(i);
        if (NORM && xn) store4(xn + (int64_t) blockIdx.x * cols + i, v);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t = fmaxf(t, red[i]);
        scale[blockIdx.x] = t / 448.0f;
        inv_s = t > 0.0f ? 448.0f / t : 0.0f;
    }
    __syncthreads();
    const float inv = inv_s;
    uint8_t* cr = codes + (int64_t) blockIdx.x * cols;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = val4(i);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z * inv, v.w * inv), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(cr + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return scale_off(rows, cols) + (size_t) rows * sizeof(float) + 16; }
float* act_scales(void* act, int64_t rows, int64_t cols) { return (float*) ((uint8_t*) act + scale_off(rows, cols)); }

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || ld % 4 != 0) {
        halo::log::error("halo", "prefill fp8: rows of %lld values (ld %lld)", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<false><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, cols, ld, codes, scale);
    ck("quantize");
}

void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || cols % 4 != 0) {
        halo::log::error("halo", "prefill fp8: rows of %lld values (norm)", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<true><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, cols, cols, codes, scale, w, eps, xn);
    ck("quantize");
}

void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, uint16_t* xn,
                       void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || cols % 4 != 0) {
        halo::log::error("halo", "prefill fp8: rows of %lld values (norm)", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<true, __nv_bfloat16><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), cols, cols, codes, scale, w, eps, reinterpret_cast<__nv_bfloat16*>(xn));
    ck("quantize");
}


// gemm_segmented's product (the checks done)
static void gemm_ada(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi, cudaStream_t stream) {
    static std::atomic<bool> raised[64];   // the shared-memory opt-in, per device
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) ck("device");
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(gemm_ada_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM) != cudaSuccess ||
            cudaFuncSetAttribute(gemm_ada_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM) != cudaSuccess) {
            halo::log::error("halo", "prefill fp8: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    AdaArgs p{};
    p.act = (const uint8_t*) act;
    int tiles = 0;
    for (int i = 0; i < nseg; ++i) {
        p.w[i] = seg[i].w;
        p.seg[i] = {seg[i].w_scale, seg[i].dst, seg[i].ld_dst, (int) seg[i].rows, tiles};
        tiles += (int) ((seg[i].rows + BR - 1) / BR);
    }
    p.a_scale = (const float*) ((const uint8_t*) act + scale_off(rows, w_cols));
    p.K = w_cols;
    p.nseg = nseg;
    p.ntok = (int) rows;
    p.tok_tiles = (int) ((rows + BT - 1) / BT);
    p.k_tiles = (int) (w_cols / A_BK);   // w_cols is a multiple of 64
    p.epi = (int) epi;
    if (epi == Epilogue::StoreBf16 || epi == Epilogue::AddBf16)
        gemm_ada_kernel<true><<<(unsigned) (tiles * p.tok_tiles), NCONS, A_SMEM, stream>>>(p);
    else
        gemm_ada_kernel<false><<<(unsigned) (tiles * p.tok_tiles), NCONS, A_SMEM, stream>>>(p);
    ck("gemm (8.9)");
}

void gemm_segmented(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi, void* stream) {
    if (rows <= 0 || nseg < 1) return;
    if (nseg > MAX_SEG || w_cols % 64 != 0) {
        halo::log::error("halo", "prefill fp8: %d segments of %lld columns", nseg, (long long) w_cols);
        std::exit(1);
    }
    const bool bf16 = epi == Epilogue::StoreBf16 || epi == Epilogue::AddBf16;
    for (int i = 0; i < nseg; ++i) {   // BF16 rows: 16-byte aligned, a multiple of 16 bytes apart
        const Segment& s = seg[i];
        if (s.rows <= 0 || s.rows % 128 != 0 || ((uintptr_t) s.w & 15) != 0 || ((uintptr_t) s.dst & (bf16 ? 15 : 7)) != 0 ||
            s.ld_dst % (bf16 ? 8 : 2) != 0 || s.ld_dst < s.rows) {
            halo::log::error("halo", "prefill fp8: segment %d: [%lld x %lld] matrix, ld %lld", i, (long long) s.rows,
                               (long long) w_cols, (long long) s.ld_dst);
            std::exit(1);
        }
    }
    gemm_ada(seg, nseg, w_cols, act, rows, epi, (cudaStream_t) stream);
}

void gemm(const uint8_t* w, int64_t w_rows, int64_t w_cols, const float* w_scale, const void* act, int64_t rows, void* dst,
          int64_t ld_dst, Epilogue epi, void* stream) {
    const Segment s{w, w_rows, w_scale, dst, ld_dst};
    gemm_segmented(&s, 1, w_cols, act, rows, epi, stream);
}

}  // namespace halo::prefill::fp8
