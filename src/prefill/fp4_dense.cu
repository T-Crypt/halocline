// src/prefill/fp4_dense.cu - see include/halo/prefill/fp4_dense.hpp.
// The dense W4A4 product on sm_89: kernel_ada, on the f16 tensor cores (see its comment).  Upstream's TMA kernel
// (adapted from Infernix's nvfp4_a4_tma_kernel, Apache-2.0, see THIRD_PARTY_LICENSES) needs compute capability 9.0 and
// was removed; the activation layout below (256-token scale tiles) is still that kernel's, and the quantizer rounds to
// E2M1 without the FP4 convert.
#include "halo/prefill/fp4_dense.hpp"

#include "halo/kernels/narrow_bf16.cuh"
#include "halo/kernels/silu.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "halo/log.hpp"


namespace halo::prefill::fp4_dense {
namespace {

constexpr int kBT = 256;                  // tokens per block
constexpr int kScaleTileBytes = kBT * 16;        // 256 tokens x 16 groups

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        halo::log::error("halo", "fp4 dense: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ------------------------------------------------------------------------------------------------- device

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }


struct Params {
    const float* gs;
    const float* ws[2];   // the matrix scales (SwiGLU: gate, up)
    float* dst;           // the plain product's output (FP32, or BF16 under BF)
    __nv_bfloat16* hdst;  // SwiGLU's output (BF16)
    int64_t ld;
    float* amax;          // SwiGLU (optional): each token's largest |output|, raised with atomicMax (zeroed by the caller)
};
// ---------------------------------------------------------------------------------------- compute capability 8.9
// kernel_ada: the W4A4 product on the f16 tensor cores.  An E2M1 code times its UE4M3 scale is exact in f16 (at most five
// significant bits, magnitudes 2^-10 to 2688), so each warp decodes both operands into f16 pairs and runs mma m16n8k16
// with fp32 accumulation: every product is the FP4 MMA's, the sums run in another order, and a token's row keeps one
// order at any chunk length.  The block tile is 128 tokens x 256 weight rows (SwiGLU: 128 gate rows and the same 128 up
// rows), eight warps as 2 (tokens) x 4 (rows) of 64 x 64, K in stages of 64 values (four, filled by cp.async from every
// thread).  Thread (g, c) of a warp takes values 16c .. 16c + 15 of each stage (one scale group) for its rows, in four mma
// steps: step s's k-slots {2c, 2c + 1} hold values 16c + 4s, +1 and {2c + 8, 2c + 9} values 16c + 4s + 2, +3, for both
// operands (window_gemm.cuh's assignment).  The epilogues are the removed TMA kernel's arithmetic, from the accumulators to
// global memory.
constexpr int kAdaBT = 128, kAdaBR = 256, kAdaStages = 4, kAdaThreads = 256;
constexpr int kAdaACodes = kAdaBT * 32, kAdaBCodes = kAdaBR * 32;               // a stage's 64 values: 32 code bytes a row
constexpr int kAdaAScales = kAdaBT * 4, kAdaBScales = kAdaBR * 4;               // and four scale bytes
constexpr int kAdaStage = kAdaACodes + kAdaBCodes + kAdaAScales + kAdaBScales;  // 13,824 bytes
constexpr int kAdaSmem = kAdaStages * kAdaStage;

struct AdaParams {
    const uint8_t* a_codes;    // [tokens][K / 2]
    const uint8_t* a_scales;   // the scale tiles (act_bytes)
    const uint8_t* w[2];       // the plane matrices (SwiGLU: gate, up)
    const float* gs;
    const float* ws[2];        // the matrix scales
    float* dst;                // the plain product's output (FP32, or BF16 under BF)
    __nv_bfloat16* hdst;       // SwiGLU's output (BF16)
    float* amax;               // SwiGLU (optional)
    int64_t ld, rows, K;
    int tokens, tok_tiles;
};

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int n) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_addr(dst)), "l"(src), "r"(n) : "memory");
}
// a UE4M3 scale byte as an f16 pair (exact)
__device__ __forceinline__ unsigned scale_f16x2(unsigned b) {
    unsigned f;
    const unsigned short v = (unsigned short) (b | (b << 8));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f;
}
// byte i of w (two E2M1 codes, the low nibble the lower half) times the f16 scale pair sc (exact)
__device__ __forceinline__ unsigned decode(unsigned w, int i, unsigned sc) {
    const unsigned f = kernels::e2m1x2_permute(w, i, 0x3e3c3800u, 0x46444240u, 0u, 0u);
    unsigned d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(f), "r"(sc));
    return d;
}
__device__ __forceinline__ void mma_f16(float* c, const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// the block's row of the warp's 8-row fragment f (SwiGLU: gate fragments 0..3, the same rows' up fragments 4..7)
template <bool SWIGLU> __device__ __forceinline__ int ada_row(int wrw, int f) {
    if constexpr (SWIGLU) return f < 4 ? 32 * wrw + 8 * f : kAdaBR / 2 + 32 * wrw + 8 * (f - 4);
    else return 64 * wrw + 8 * f;
}

template <bool SWIGLU, bool ADD, bool BF = false>
__global__ __launch_bounds__(kAdaThreads, 1) void kernel_ada(const __grid_constant__ AdaParams p) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int loaded = SWIGLU ? kAdaBR / 2 : kAdaBR;   // the rows a block reads of each matrix
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
    const int wtk = warp >> 2, wrw = warp & 3;
    const int tok_tile = (int) blockIdx.x % p.tok_tiles;
    const int64_t row_begin = (int64_t) ((int) blockIdx.x / p.tok_tiles) * loaded;
    const int token_begin = tok_tile * kAdaBT;
    const int64_t K = p.K, half = K / 2;
    const int ktiles = (int) (K / 64);
    const size_t scale_off = (size_t) p.rows * (size_t) half;   // nvfp4p::scale_offset
    // stage st <- K tile kt (values 64 kt ..): codes and scales, zeros past the tokens and the rows
    const auto load = [&](int kt, int st) {
        unsigned char* ac = smem + st * kAdaStage;
        unsigned char* bc = ac + kAdaACodes;
        unsigned char* asc = bc + kAdaBCodes;
        unsigned char* bsc = asc + kAdaAScales;
        {
            const int r = tid >> 1, pc = tid & 1, t = token_begin + r;
            const bool ok = t < p.tokens;
            cp_async16(ac + r * 32 + pc * 16, p.a_codes + (int64_t) (ok ? t : 0) * half + kt * 32 + pc * 16, ok ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * kAdaThreads, r = idx >> 1, pc = idx & 1;
            const int m = SWIGLU ? r / loaded : 0;
            const int64_t wr = row_begin + (SWIGLU ? r % loaded : r);
            const bool ok = wr < p.rows;
            cp_async16(bc + r * 32 + pc * 16, p.w[m] + (ok ? wr : 0) * half + kt * 32 + pc * 16, ok ? 16 : 0);
        }
        if (tid < kAdaBT) {   // a token's four groups: scale tile (t / 256) * (K / 256) + kt / 4, row t % 256
            const int t = token_begin + tid;
            const bool ok = t < p.tokens;
            const int64_t tt = ok ? t : 0;
            const uint8_t* src = p.a_scales + ((tt / kBT) * (K / 256) + kt / 4) * kScaleTileBytes + (tt % kBT) * 16 + (kt % 4) * 4;
            cp_async4(asc + tid * 4, src, ok ? 4 : 0);
        }
        {   // a row's four groups: nvfp4p::scale_index(row, 4 kt .. 4 kt + 3, K)
            const int r = tid;
            const int m = SWIGLU ? r / loaded : 0;
            const int64_t wr = row_begin + (SWIGLU ? r % loaded : r);
            const bool ok = wr < p.rows;
            const int64_t rr = ok ? wr : 0;
            const uint8_t* src = p.w[m] + scale_off + ((rr / 128) * (K / 64) + kt) * 512 + (rr & 31) * 16 + ((rr & 127) >> 5) * 4;
            cp_async4(bsc + r * 4, src, ok ? 4 : 0);
        }
    };
    float acc[4][8][4] = {};
#pragma unroll
    for (int st = 0; st < kAdaStages - 1; ++st) {
        if (st < ktiles) load(st, st);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll 1
    for (int kt = 0; kt < ktiles; ++kt) {
        asm volatile("cp.async.wait_group %0;" ::"n"(kAdaStages - 2) : "memory");
        __syncthreads();   // stage kt is in, and every warp has left stage kt - 1, which the next load refills
        if (kt + kAdaStages - 1 < ktiles) load(kt + kAdaStages - 1, (kt + kAdaStages - 1) % kAdaStages);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const unsigned char* ac = smem + (kt % kAdaStages) * kAdaStage;
        const unsigned char* bc = ac + kAdaACodes;
        const unsigned char* asc = bc + kAdaBCodes;
        const unsigned char* bsc = asc + kAdaAScales;
        // the thread's 8 code bytes (values 16c .. 16c + 15) and scale of each of its rows
        uint2 ar[4][2], br[8];
        unsigned as2[4][2], bs2[8];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wtk * 64 + mt * 16 + g + 8 * h;
                ar[mt][h] = *reinterpret_cast<const uint2*>(ac + r * 32 + 8 * c);
                as2[mt][h] = scale_f16x2(asc[r * 4 + c]);
            }
#pragma unroll
        for (int f = 0; f < 8; ++f) {
            const int r = ada_row<SWIGLU>(wrw, f) + g;
            br[f] = *reinterpret_cast<const uint2*>(bc + r * 32 + 8 * c);
            bs2[f] = scale_f16x2(bsc[r * 4 + c]);
        }
#pragma unroll
        for (int step = 0; step < 4; ++step) {   // bytes 2 step and 2 step + 1 of the eight
            const int bi = (step & 1) * 2;
            unsigned a[4][4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const unsigned w0 = step >> 1 ? ar[mt][0].y : ar[mt][0].x, w1 = step >> 1 ? ar[mt][1].y : ar[mt][1].x;
                a[mt][0] = decode(w0, bi, as2[mt][0]);       // row g: values 16c + 4 step, +1
                a[mt][1] = decode(w1, bi, as2[mt][1]);       // row g + 8
                a[mt][2] = decode(w0, bi + 1, as2[mt][0]);   // row g: values 16c + 4 step + 2, +3
                a[mt][3] = decode(w1, bi + 1, as2[mt][1]);
            }
#pragma unroll
            for (int f = 0; f < 8; ++f) {   // each weight fragment decoded right before its four products
                const unsigned w = step >> 1 ? br[f].y : br[f].x;
                const unsigned b0 = decode(w, bi, bs2[f]), b1 = decode(w, bi + 1, bs2[f]);
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) mma_f16(acc[mt][f], a[mt], b0, b1);
            }
        }
    }
    // kernel's epilogue arithmetic: thread (g, c) holds tokens g and g + 8 of each 16-token fragment, rows 2c and 2c + 1
    // of each 8-row fragment; the scale a rounded multiply, the add after it a separate rounded add
    const float s0 = *p.ws[0];
    const float s1 = SWIGLU ? *p.ws[1] : 0.0f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = token_begin + wtk * 64 + mt * 16 + g + 8 * h;
            const bool live = t < p.tokens;
            const float gt = live ? p.gs[t] : 0.0f;
            if constexpr (SWIGLU) {
                const float ag = __fmul_rn(gt, s0), au = __fmul_rn(gt, s1);
                float m = 0.0f;
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    float v[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float gg = __fmul_rn(acc[mt][f][2 * h + e], ag);
                        const float uu = __fmul_rn(acc[mt][f + 4][2 * h + e], au);
                        v[e] = __fmul_rn(kernels::silu_quotient(gg), uu);
                    }
                    const __nv_bfloat162 hv = __floats2bfloat162_rn(v[0], v[1]);   // the amax is of the stored values
                    m = fmaxf(m, fmaxf(fabsf(__low2float(hv)), fabsf(__high2float(hv))));
                    if (live) *reinterpret_cast<__nv_bfloat162*>(p.hdst + (int64_t) t * p.ld + row_begin + 32 * wrw + 8 * f + 2 * c) = hv;
                }
                if (p.amax) {   // the token's largest |value|: its four lanes, then one atomic (max is exact)
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    if (c == 0 && live) atomicMax(reinterpret_cast<int*>(p.amax + t), __float_as_int(m));
                }
            } else {
                if (!live) continue;
                const float alpha = __fmul_rn(gt, s0);
                float2 res[8];   // ADD: every residual pair loaded before any is written back
#pragma unroll
                for (int f = 0; f < 8; ++f) {
                    const int64_t r = row_begin + 64 * wrw + 8 * f + 2 * c;
                    res[f] = make_float2(0.0f, 0.0f);
                    if (ADD && r < p.rows) {
                        if constexpr (BF)
                            res[f] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                                reinterpret_cast<const __nv_bfloat16*>(p.dst) + (int64_t) t * p.ld + r));
                        else
                            res[f] = __ldcg(reinterpret_cast<const float2*>(p.dst + (int64_t) t * p.ld + r));
                    }
                }
#pragma unroll
                for (int f = 0; f < 8; ++f) {
                    const int64_t r = row_begin + 64 * wrw + 8 * f + 2 * c;
                    if (r >= p.rows) continue;
                    float2 v = make_float2(__fmul_rn(acc[mt][f][2 * h], alpha), __fmul_rn(acc[mt][f][2 * h + 1], alpha));
                    if constexpr (ADD) v = make_float2(__fadd_rn(res[f].x, v.x), __fadd_rn(res[f].y, v.y));
                    if constexpr (BF)
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.dst) + (int64_t) t * p.ld + r) =
                            __floats2bfloat162_rn(v.x, v.y);
                    else
                        *reinterpret_cast<float2*>(p.dst + (int64_t) t * p.ld + r) = v;
                }
            }
        }
}

// E2M1 to nearest, ties to the even code, saturating: the rounding of cvt.rn.satfinite.e2m1x2.f32, which 8.9 lacks
// (nvfp4_planes.cu rounds the same way)
__device__ __forceinline__ unsigned e2m1_rn(float x) {
    const float a = fabsf(x);
    const unsigned c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
    return c | (x < 0.0f && c ? 8u : 0u);
}

// One warp per token row (and per padding row of the last scale tile).  Token scale g = amax / (6 * 448); per 16 values
// a UE4M3 scale of block amax / (6 g); codes round to nearest even on x / (scale * g), value 2b in byte b's low nibble.
// NORM: the row is RMSNorm'd first (x * rsqrt(mean(x^2) + eps) * nw), the normed row optionally written to xn (Xn: FP32,
// or rounded to BF16).
// AMAX: each row's largest |value| is given (rowmax), so the row is read once.  In: float or __nv_bfloat16 rows.
template <bool NORM, bool AMAX = false, class In = float, class Xn = float>
__global__ void quantize_kernel(const In* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                float* __restrict__ gs, int cols, int64_t ld, int64_t rows, const float* __restrict__ nw = nullptr,
                                float eps = 0.0f, Xn* __restrict__ xn = nullptr, const float* __restrict__ rowmax = nullptr) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nb = cols / 16, ktiles = cols / 256;
    const int64_t tile_base = (r / kBT) * ktiles;
    uint8_t* srow = scales + (r % kBT) * 16;
    if (r >= rows) {   // padding of the last token tile: its scales are defined (zero), no codes are read
        for (int kt = lane; kt < ktiles; kt += 32) *reinterpret_cast<uint4*>(srow + (tile_base + kt) * kScaleTileBytes) = make_uint4(0, 0, 0, 0);
        return;
    }
    const In* xr = x + r * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 v;
                if constexpr (std::is_same_v<In, float>) {
                    v = *(const float4*) (xr + 16 * b + 4 * i);
                } else {
                    const uint2 u = *(const uint2*) (xr + 16 * b + 4 * i);
                    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                    v = make_float4(a.x, a.y, c.x, c.y);
                }
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        inv_rms = rsqrtf(ss / (float) cols + eps);
    }
    // value c of the row as the quantizer sees it
    auto val4 = [&](int c) -> float4 {
        float4 v;
        if constexpr (std::is_same_v<In, float>) {
            v = *(const float4*) (xr + c);
        } else {
            const uint2 u = *(const uint2*) (xr + c);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            v = make_float4(a.x, a.y, b.x, b.y);
        }
        if (NORM) {
            const float4 w = *(const float4*) (nw + c);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float amax = 0.0f;
    if (AMAX) {
        amax = rowmax[r];
    } else {
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = val4(16 * b + 4 * i);
                if (NORM && xn) {
                    if constexpr (std::is_same_v<Xn, float>) {
                        *(float4*) (xn + r * cols + 16 * b + 4 * i) = v;
                    } else {
                        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
                        const __nv_bfloat162 c = __floats2bfloat162_rn(v.z, v.w);
                        *reinterpret_cast<uint2*>(xn + r * cols + 16 * b + 4 * i) =
                            make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&c));
                    }
                }
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) gs[r] = g;
    const float inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float v[16];
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 t = val4(16 * b + 4 * i);
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(m * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
        uint32_t w[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short c;
            c = (unsigned short) (e2m1_rn(v[2 * j] * inv) | (e2m1_rn(v[2 * j + 1] * inv) << 4));
            w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
        }
        *reinterpret_cast<uint2*>(codes + r * (cols / 2) + 8 * b) = make_uint2(w[0], w[1]);
        srow[(tile_base + b / 16) * kScaleTileBytes + (b & 15)] = (uint8_t) sb;
    }
}

// ----------------------------------------------------------------------------------------------------- host

int64_t padded(int64_t tokens) { return (tokens + kBT - 1) / kBT * kBT; }

template <bool SWIGLU, bool ADD, bool BF>
void launch_ada(const Params& q, const void* w0, const void* w1, int64_t rows, int64_t K, const void* act, int64_t tokens,
                cudaStream_t s) {
    constexpr int loaded = SWIGLU ? kAdaBR / 2 : kAdaBR;
    static std::atomic<bool> configured[64];   // the shared-memory opt-in is per device and per kernel
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(kernel_ada<SWIGLU, ADD, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAdaSmem), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    AdaParams p{};
    p.a_codes = (const uint8_t*) act;
    p.a_scales = (const uint8_t*) act + tokens * (K / 2);
    p.w[0] = (const uint8_t*) w0;
    p.w[1] = (const uint8_t*) (SWIGLU ? w1 : w0);
    p.gs = q.gs;
    p.ws[0] = q.ws[0];
    p.ws[1] = q.ws[1];
    p.dst = q.dst;
    p.hdst = q.hdst;
    p.amax = q.amax;
    p.ld = q.ld;
    p.rows = rows;
    p.K = K;
    p.tokens = (int) tokens;
    p.tok_tiles = (int) ((tokens + kAdaBT - 1) / kAdaBT);
    const int64_t row_tiles = (rows + loaded - 1) / loaded;
    kernel_ada<SWIGLU, ADD, BF><<<(unsigned) (row_tiles * p.tok_tiles), kAdaThreads, kAdaSmem, s>>>(p);
    ck(cudaGetLastError(), "product (8.9)");
}

template <bool SWIGLU, bool ADD, bool BF = false> void launch(Params p, const void* w0, const void* w1, int64_t rows, int64_t K,
                                             const void* act, int64_t tokens, cudaStream_t s) {
    launch_ada<SWIGLU, ADD, BF>(p, w0, w1, rows, K, act, tokens, s);
}

// `bf16`: the output rows are BF16, written by tensor stores (16-byte aligned, a multiple of 16 bytes apart).
void check(const void* a, const void* b, int64_t rows, int64_t K, int64_t tokens, const void* act, const void* dst, int64_t ld,
           const char* what, bool bf16 = false, const float* gs = nullptr) {
    const auto al = [](const void* v, size_t n) { return (reinterpret_cast<uintptr_t>(v) & (n - 1)) == 0; };
    if (rows <= 0 || rows % 128 || K <= 0 || K % 256 || tokens <= 0 || ld < rows || (ld & (bf16 ? 7 : 1)) || !al(a, 16) || !al(b, 16) ||
        !al(act, 16) || !al(dst, bf16 ? 16 : 8) || (bf16 && !al(gs, 16))) {
        halo::log::error("halo", "fp4 dense: %s: rows %lld K %lld tokens %lld ld %lld", what, (long long) rows, (long long) K,
                           (long long) tokens, (long long) ld);
        std::exit(1);
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 2) + (size_t) padded(rows) * (size_t) (cols / 16);
}

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        halo::log::error("halo", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, ld, rows);
    ck(cudaGetLastError(), "quantize");
}

void quantize_amax(const uint16_t* x, const float* rowmax, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs,
                   void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        halo::log::error("halo", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false, true, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, ld, rows, nullptr, 0.0f,
                                                                          static_cast<float*>(nullptr), rowmax);
    ck(cudaGetLastError(), "quantize_amax");
}

void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs, float* xn,
                       void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        halo::log::error("halo", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, cols, rows, w, eps, xn);
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void product(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs, int64_t tokens,
             float* dst, int64_t ld, bool add, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product");
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = dst;
    p.ld = ld;
    if (add) launch<false, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
    else launch<false, false>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void product_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                  int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_bf16", true, gs);
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = reinterpret_cast<float*>(dst);
    p.ld = ld;
    launch<false, false, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void product_add_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                      int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_add_bf16", true, gs);
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = reinterpret_cast<float*>(dst);
    p.ld = ld;
    launch<false, true, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs,
                       uint16_t* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15) ||
        (xn && (reinterpret_cast<uintptr_t>(xn) & 15))) {
        halo::log::error("halo", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true, false, __nv_bfloat16, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, cols, rows, w, eps,
        reinterpret_cast<__nv_bfloat16*>(xn));
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void swiglu(const void* w_gate, const void* w_up, const float* s_gate, const float* s_up, int64_t rows, int64_t K,
            const void* act, const float* gs, int64_t tokens, uint16_t* dst, int64_t ld, void* stream, float* amax) {
    check(w_gate, w_up, rows, K, tokens, act, dst, ld, "swiglu", true, gs);
    Params p{};
    p.amax = amax;
    if (amax && cudaMemsetAsync(amax, 0, (size_t) tokens * sizeof(float), (cudaStream_t) stream) != cudaSuccess)
        ck(cudaGetLastError(), "swiglu amax");
    p.gs = gs;
    p.ws[0] = s_gate;
    p.ws[1] = s_up;
    p.hdst = reinterpret_cast<__nv_bfloat16*>(dst);
    p.ld = ld;
    launch<true, false>(p, w_gate, w_up, rows, K, act, tokens, (cudaStream_t) stream);
}

}  // namespace halo::prefill::fp4_dense
