// src/models/inkling2/kernels/ink_ops.cu - the non-GEMM kernels of Inkling 2 Mini on sm_89. See ink_ops.hpp.
// clip_rows, rms_rows, loop_control and short_conv started from a local-model draft (reviewed); the attention, the
// depth mix, the MoE routing and the small ones are written for first light: correct first, then fast.
#include "models/inkling2/kernels/ink_ops.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace halo::models::inkling2::k {
namespace {

unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the sum of v over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

__device__ __forceinline__ float bf16_bits(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// ---- clip_rows: one warp per row (width <= 128, a multiple of 4) ----
__global__ void clip_rows_kernel(float* x, const float* __restrict__ scale, int scale_rows, int64_t rows, int width,
                                 float threshold) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31, n4 = width >> 2;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (lane < n4) {
        v = *reinterpret_cast<const float4*>(x + row * width + 4 * lane);
        if (scale) {
            const float4 s4 = *reinterpret_cast<const float4*>(scale + (row % scale_rows) * width + 4 * lane);
            v.x *= s4.x, v.y *= s4.y, v.z *= s4.z, v.w *= s4.w;
        }
    }
    const float n = sqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w)) + 1e-6f;
    const float c = fminf(1.0f, threshold / n);
    if (lane < n4) *reinterpret_cast<float4*>(x + row * width + 4 * lane) = make_float4(v.x * c, v.y * c, v.z * c, v.w * c);
}

// ---- rms_rows: one warp per row of width <= 256, else one block per row ----
__global__ void rms_rows_warp(const float* x, const float* __restrict__ w, float* out, int64_t rows, int width, float eps) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* xr = x + row * width;
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = lane + 32 * i;
        v[i] = c < width ? xr[c] : 0.f;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(warp_sum(ss) / (float) width + eps);
    float* yr = out + row * width;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = lane + 32 * i;
        if (c < width) yr[c] = w ? v[i] * inv * w[c] : v[i] * inv;
    }
}
__global__ void rms_rows_block(const float* x, const float* __restrict__ w, float* out, int width, float eps) {
    __shared__ float red[32];
    const float* xr = x + (int64_t) blockIdx.x * width;
    float* yr = out + (int64_t) blockIdx.x * width;
    float ss = 0.f;
    for (int c = threadIdx.x; c < width; c += blockDim.x) ss += xr[c] * xr[c];
    const float inv = rsqrtf(block_sum(ss, red) / (float) width + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x) yr[c] = w ? xr[c] * inv * w[c] : xr[c] * inv;
}

// ---- depth_mix: one block per row ----
struct StatePtrs {
    const float* p[16];
};
__global__ void depth_mix_kernel(StatePtrs st, int n, const float* __restrict__ query, int nq, float* __restrict__ out, int D) {
    __shared__ float red[32];
    __shared__ float inv[16], prob[3][16];
    const int64_t t = blockIdx.x;
    float dots[3][16];
    for (int j = 0; j < n; ++j) {
        const float* s = st.p[j] + t * D;
        float ss = 0.f, d0 = 0.f, d1 = 0.f, d2 = 0.f;
        for (int c = threadIdx.x; c < D; c += blockDim.x) {
            const float v = s[c];
            ss += v * v;
            d0 += v * query[c];
            if (nq > 1) d1 += v * query[D + c];
            if (nq > 2) d2 += v * query[2 * D + c];
        }
        const float tss = block_sum(ss, red);
        const float iv = rsqrtf(tss / (float) D + 1e-6f);
        dots[0][j] = block_sum(d0, red) * iv;
        dots[1][j] = nq > 1 ? block_sum(d1, red) * iv : 0.f;
        dots[2][j] = nq > 2 ? block_sum(d2, red) * iv : 0.f;
        if (threadIdx.x == 0) inv[j] = iv;
    }
    if (threadIdx.x == 0) {
        const float sc = rsqrtf((float) D);
        for (int qi = 0; qi < nq; ++qi) {
            float m = -INFINITY;
            for (int j = 0; j < n; ++j) m = fmaxf(m, dots[qi][j] * sc);
            float l = 0.f;
            for (int j = 0; j < n; ++j) l += prob[qi][j] = expf(dots[qi][j] * sc - m);
            for (int j = 0; j < n; ++j) prob[qi][j] /= l;
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        for (int j = 0; j < n; ++j) {
            const float v = st.p[j][t * D + c] * inv[j];
            a0 += prob[0][j] * v;
            if (nq > 1) a1 += prob[1][j] * v;
            if (nq > 2) a2 += prob[2][j] * v;
        }
        float* o = out + t * (int64_t) nq * D;
        o[c] = a0;
        if (nq > 1) o[D + c] = a1;
        if (nq > 2) o[2 * D + c] = a2;
    }
}

// ---- loop_control: one block of 256 per row ----
constexpr int kLcMaxD = 2304, kLcMaxRank = 64;
__global__ void loop_control_kernel(const float* cur, const float* msg, const float* __restrict__ w_down,
                                    const float* __restrict__ w_gain, const float* __restrict__ w_forget, float* out,
                                    int D, int rank) {
    __shared__ float hc[kLcMaxD], hm[kLcMaxD], z[kLcMaxRank], red[32];
    const int64_t t = blockIdx.x;
    const float* cr = cur + t * D;
    const float* mr = msg + t * D;
    float sc = 0.f, sm = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float a = cr[i], b = mr[i];
        hc[i] = a, hm[i] = b;
        sc += a * a, sm += b * b;
    }
    const float ic = rsqrtf(block_sum(sc, red) / (float) D + 1e-6f);
    const float im = rsqrtf(block_sum(sm, red) / (float) D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) hc[i] *= ic, hm[i] *= im;
    __syncthreads();
    // z: a warp per row of w_down, looping
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = warp; r < rank; r += nw) {
        const float* wr = w_down + (int64_t) r * 2 * D;
        float a = 0.f;
        for (int i = lane; i < D; i += 32) a += wr[i] * hc[i] + wr[D + i] * hm[i];
        a = warp_sum(a);
        if (lane == 0) z[r] = a;
    }
    __syncthreads();
    float* orow = out + t * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float g = 0.f, f = 0.f;
        const float* wg = w_gain + (int64_t) i * rank;
        const float* wf = w_forget + (int64_t) i * rank;
        for (int r = 0; r < rank; ++r) g += z[r] * wg[r], f += z[r] * wf[r];
        g = tanhf(g);
        f = tanhf(f);
        orow[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, f), cr[i]), __fmul_rn(g, mr[i]));
    }
}

// ---- short_conv: one thread per channel, the rows in order ----
__global__ void short_conv_kernel(float* x, int T, int C, const float* __restrict__ w, const float* __restrict__ dyn,
                                  int n_heads, float* state) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int h = n_heads > 0 ? c / (C / n_heads) : 0;
    float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const float sc = 1.0f / sqrtf((dyn ? 2.f : 1.f) * 4.f + 1.f);
    for (int t = 0; t < T; ++t) {
        const float xt = x[(int64_t) t * C + c];
        float o = xt + s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        if (dyn) {
            const float* d = dyn + ((int64_t) t * n_heads + h) * 4;
            o += xt * d[0] + s2 * d[1] + s1 * d[2] + s0 * d[3];
        }
        x[(int64_t) t * C + c] = o * sc;
        s0 = s1, s1 = s2, s2 = xt;
    }
    state[c * 3 + 0] = s0, state[c * 3 + 1] = s1, state[c * 3 + 2] = s2;
}

// ---- attention ----
constexpr int kNh = 24, kNkv = 8, kGroup = 3, kHd = 128, kRel = 16;
constexpr int kAttnWarps = 4;

struct Split {
    int splits;
    int64_t slice;
};
Split plan_split(int T, int64_t keys) {
    const int64_t want = std::max<int64_t>(1, (1024 + (int64_t) T * kNkv - 1) / ((int64_t) T * kNkv));
    const int64_t most = std::max<int64_t>(1, (keys + 255) / 256);
    const int splits = (int) std::min(want, most);
    return {splits, (keys + splits - 1) / splits};
}

// One block of 4 warps per (row t, KV head g, split): every warp takes keys of the split in turn and keeps an online
// softmax for the group's 3 query heads (4 values of each head per lane); the warps merge at the end and the block
// writes (m, l, acc[128]) per head. A key's score is one warp sum per head: the lanes' q.k parts (scaled) plus, on
// lanes 0..15, r . relb[dist] parts.
__global__ void attn_split_kernel(AttnArgs a, int splits, int64_t slice, float* __restrict__ ws) {
    const int tg = blockIdx.x, sp = blockIdx.y;
    const int t = tg / kNkv, g = tg % kNkv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t pt = a.pos0 + t;
    const int64_t lo = a.window > 0 ? (pt - a.window + 1 > 0 ? pt - a.window + 1 : 0) : 0;
    const int64_t j0 = lo + (int64_t) sp * slice, j1 = (pt + 1 < j0 + slice) ? pt + 1 : j0 + slice;
    float4 q[kGroup];
    float r[kGroup];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
        const int h = g * kGroup + j;
        q[j] = *reinterpret_cast<const float4*>(a.q + ((int64_t) t * kNh + h) * kHd + lane * 4);
        q[j].x *= a.scale, q[j].y *= a.scale, q[j].z *= a.scale, q[j].w *= a.scale;
        r[j] = lane < kRel ? a.r[((int64_t) t * kNh + h) * kRel + lane] : 0.f;
    }
    float m[kGroup], l[kGroup];
    float4 acc[kGroup];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) m[j] = -INFINITY, l[j] = 0.f, acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int64_t p = j0 + warp; p < j1; p += kAttnWarps) {
        const int64_t cell = a.ring ? p % a.ring : p;
        const uint2 kb = *reinterpret_cast<const uint2*>(a.kc + (cell * kNkv + g) * kHd + lane * 4);
        const uint2 vb = *reinterpret_cast<const uint2*>(a.vc + (cell * kNkv + g) * kHd + lane * 4);
        const float k0 = bf16_bits((uint16_t) kb.x), k1 = bf16_bits((uint16_t) (kb.x >> 16));
        const float k2 = bf16_bits((uint16_t) kb.y), k3 = bf16_bits((uint16_t) (kb.y >> 16));
        const float v0 = bf16_bits((uint16_t) vb.x), v1 = bf16_bits((uint16_t) (vb.x >> 16));
        const float v2 = bf16_bits((uint16_t) vb.y), v3 = bf16_bits((uint16_t) (vb.y >> 16));
        const int64_t dist = pt - p;
        const float rb = (lane < kRel && dist < a.extent) ? a.relb[dist * kRel + lane] : 0.f;
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
            const float s = warp_sum(q[j].x * k0 + q[j].y * k1 + q[j].z * k2 + q[j].w * k3 + r[j] * rb);
            const float mn = fmaxf(m[j], s);
            const float al = __expf(m[j] - mn), e = __expf(s - mn);
            m[j] = mn;
            l[j] = l[j] * al + e;
            acc[j].x = acc[j].x * al + e * v0;
            acc[j].y = acc[j].y * al + e * v1;
            acc[j].z = acc[j].z * al + e * v2;
            acc[j].w = acc[j].w * al + e * v3;
        }
    }
    // merge the warps
    __shared__ float sm_m[kAttnWarps][kGroup], sm_l[kAttnWarps][kGroup];
    __shared__ float4 sm_acc[kAttnWarps][kGroup][32];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
        if (lane == 0) sm_m[warp][j] = m[j], sm_l[warp][j] = l[j];
        sm_acc[warp][j][lane] = acc[j];
    }
    __syncthreads();
    if (warp < kGroup) {
        const int j = warp, h = g * kGroup + j;
        float M = -INFINITY;
        for (int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, sm_m[w][j]);
        float L = 0.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int w = 0; w < kAttnWarps; ++w) {
            const float f = M == -INFINITY ? 0.f : __expf(sm_m[w][j] - M);
            L += sm_l[w][j] * f;
            const float4 x = sm_acc[w][j][lane];
            A.x += x.x * f, A.y += x.y * f, A.z += x.z * f, A.w += x.w * f;
        }
        float* o = ws + (((int64_t) t * kNh + h) * splits + sp) * (kHd + 2);
        if (lane == 0) o[0] = M, o[1] = L;
        *reinterpret_cast<float2*>(o + 2 + lane * 4) = make_float2(A.x, A.y);
        *reinterpret_cast<float2*>(o + 2 + lane * 4 + 2) = make_float2(A.z, A.w);
    }
}

// One warp per (row, head): the splits merged with the sink logit in the denominator.
__global__ void attn_combine_kernel(AttnArgs a, int splits, const float* __restrict__ ws) {
    const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t) a.T * kNh) return;
    const float* base = ws + wid * splits * (kHd + 2);
    const float sink = a.sink[wid];
    float M = sink;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (kHd + 2)]);
    float L = __expf(sink - M);
    float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < splits; ++s) {
        const float* o = base + s * (kHd + 2);
        const float f = o[0] == -INFINITY ? 0.f : __expf(o[0] - M);
        L += o[1] * f;
        const float2 x = *reinterpret_cast<const float2*>(o + 2 + lane * 4);
        const float2 y = *reinterpret_cast<const float2*>(o + 2 + lane * 4 + 2);
        A.x += x.x * f, A.y += x.y * f, A.z += y.x * f, A.w += y.y * f;
    }
    const float inv = 1.0f / L;
    *reinterpret_cast<float4*>(a.out + wid * kHd + lane * 4) = make_float4(A.x * inv, A.y * inv, A.z * inv, A.w * inv);
}

// ---- MoE ----
// one warp per token: p = sigmoid, k rounds of a warp argmax of p + bias
__global__ void moe_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T, int E, int K,
                                 float w_scale, int32_t* sel, float* weight) {
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    constexpr int kPer = 8;   // E <= 256
    float p[kPer], s[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e = lane + 32 * i;
        p[i] = e < E ? 1.0f / (1.0f + __expf(-logits[(int64_t) t * E + e])) : 0.f;
        s[i] = e < E ? p[i] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    float sum = 0.f;
    float wsel[16];
    int isel[16];
    for (int r = 0; r < K; ++r) {
        float best = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int e = lane + 32 * i;
            if (s[i] > best || (s[i] == best && e < bi)) best = s[i], bi = e;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
        }
        // the owner clears it and publishes its p
        float pv = 0.f;
#pragma unroll
        for (int i = 0; i < kPer; ++i)
            if (lane + 32 * i == bi) pv = p[i], s[i] = -INFINITY;
        pv = warp_sum(pv);
        isel[r] = bi;
        wsel[r] = pv;
        sum += pv;
    }
    if (lane == 0) {
        const float d = sum + 1e-20f;
        for (int r = 0; r < K; ++r) {
            sel[(int64_t) t * K + r] = isel[r];
            weight[(int64_t) t * K + r] = wsel[r] / d * w_scale;
        }
    }
}

// one warp per expert: its picks in (token, slot) order by ballots; pass 0 counts, pass 1 places
__global__ void moe_count_kernel(const int32_t* __restrict__ sel, int R, int E, int32_t* counts) {
    const int e = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (e >= E) return;
    int c = 0;
    for (int i0 = 0; i0 < R; i0 += 32) {
        const int i = i0 + lane;
        c += __popc(__ballot_sync(0xffffffffu, i < R && sel[i] == e));
    }
    if (lane == 0) counts[e] = c;
}
__global__ void moe_scan_kernel(const int32_t* counts, int E, int32_t* bounds) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < E; ++e) {
            const int c = counts[e];   // counts may be bounds itself: read before the write
            bounds[e] = acc;
            acc += c;
        }
        bounds[E] = acc;
    }
}
__global__ void moe_place_kernel(const int32_t* __restrict__ sel, int R, int K, int E, const int32_t* __restrict__ bounds,
                                 int32_t* row_of, int32_t* src_token, int32_t* expert_of) {
    const int e = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (e >= E) return;
    int base = bounds[e];
    for (int i0 = 0; i0 < R; i0 += 32) {
        const int i = i0 + lane;
        const bool mine = i < R && sel[i] == e;
        const unsigned b = __ballot_sync(0xffffffffu, mine);
        if (mine) {
            const int row = base + __popc(b & ((1u << lane) - 1u));
            row_of[i] = row;
            src_token[row] = i / K;
            expert_of[row] = e;
        }
        base += __popc(b);
    }
}

__global__ void moe_situ_kernel(const float* __restrict__ gu, const int32_t* __restrict__ expert_of,
                                const float* __restrict__ gu_scale, float* __restrict__ h, int R, int F, float beta,
                                float lbeta) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) R * F) return;
    const int64_t row = i / F, c = i % F;
    const float sc = gu_scale ? gu_scale[expert_of[row]] : 1.f;
    const float g = gu[row * 2 * F + c] * sc, u = gu[row * 2 * F + F + c] * sc;
    const float act = beta * tanhf(g / beta) * (1.f / (1.f + __expf(-g)));
    h[i] = act * (lbeta > 0.f ? lbeta * tanhf(u / lbeta) : u);
}

__global__ void moe_combine_kernel(const float* __restrict__ down, const int32_t* __restrict__ row_of,
                                   const int32_t* __restrict__ sel, const float* __restrict__ weight,
                                   const float* __restrict__ d_scale, float* __restrict__ out, int T, int K, int D) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * D) return;
    const int64_t t = i / D, c = i % D;
    float a = 0.f;
    for (int s = 0; s < K; ++s) {
        const int64_t pk = t * K + s;
        const float w = weight[pk] * (d_scale ? d_scale[sel[pk]] : 1.f);
        a += w * down[(int64_t) row_of[pk] * D + c];
    }
    out[i] = a;
}

// ---- small ----
__global__ void log_tau_kernel(int64_t pos0, int T, float floor, float alpha, float* tau) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) tau[t] = 1.f + alpha * logf(fmaxf(1.f, (float) (pos0 + t + 1) / floor));
}
__global__ void scale_heads_kernel(float* x, const float* __restrict__ tau, int T, int per_row) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) T * per_row) x[i] *= tau[i / per_row];
}
__global__ void sigmoid_gate_kernel(float* o, const float* __restrict__ g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= 1.f / (1.f + __expf(-g[i]));
}
__global__ void softcap_kernel(float* x, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
__global__ void kv_store_kernel(const float* __restrict__ k, const float* __restrict__ v, uint16_t* kc, uint16_t* vc,
                                int64_t pos0, int T, int64_t ring) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) kNkv * kHd;
    if (i >= (int64_t) T * per) return;
    const int64_t t = i / per, w = i % per, p = pos0 + t, cell = ring ? p % ring : p;
    const __nv_bfloat16 kb = __float2bfloat16_rn(k[i]), vb = __float2bfloat16_rn(v[i]);
    kc[cell * per + w] = *reinterpret_cast<const uint16_t*>(&kb);
    vc[cell * per + w] = *reinterpret_cast<const uint16_t*>(&vb);
}
__global__ void add_rows_kernel(float* x, const float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

}  // namespace

void clip_rows(float* x, const float* scale, int scale_rows, int64_t rows, int width, float threshold, void* stream) {
    if (rows <= 0) return;
    clip_rows_kernel<<<blocks_for(rows * 32), 256, 0, (cudaStream_t) stream>>>(x, scale, scale_rows > 0 ? scale_rows : 1, rows,
                                                                              width, threshold);
}

void rms_rows(const float* x, const float* w, float* out, int64_t rows, int width, float eps, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    if (width <= 256) rms_rows_warp<<<blocks_for(rows * 32), 256, 0, s>>>(x, w, out, rows, width, eps);
    else rms_rows_block<<<(unsigned) rows, 256, 0, s>>>(x, w, out, width, eps);
}

void depth_mix(const float* const* states, int n_states, const float* query, int n_query, float* out, int T, int D,
               void* stream) {
    StatePtrs p{};
    for (int j = 0; j < n_states && j < 16; ++j) p.p[j] = states[j];
    depth_mix_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(p, std::min(n_states, 16), query, n_query, out, D);
}

void loop_control(const float* cur, const float* msg, const float* w_down, const float* w_gain, const float* w_forget,
                  float* out, int T, int D, int rank, void* stream) {
    loop_control_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(cur, msg, w_down, w_gain, w_forget, out, D, rank);
}

void short_conv(float* x, int T, int C, const float* w, const float* dyn, int n_heads, float* state, void* stream) {
    short_conv_kernel<<<blocks_for(C, 128), 128, 0, (cudaStream_t) stream>>>(x, T, C, w, dyn, n_heads, state);
}

size_t ink_attention_workspace(int T, int64_t n_keys_max) {
    // the most (rows x splits) any call of up to T rows over up to n_keys_max keys takes
    int64_t most = 0;
    for (int t = 1; t <= T; ++t) most = std::max<int64_t>(most, (int64_t) t * plan_split(t, n_keys_max).splits);
    return (size_t) most * kNh * (kHd + 2) * sizeof(float);
}

void ink_attention(const AttnArgs& a, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    const int64_t last = a.pos0 + a.T;   // keys of the last row
    const int64_t keys = a.window > 0 ? std::min<int64_t>(a.window, last) : last;
    Split sp = plan_split(a.T, keys);
    // the splits are fixed by the workspace: shrink them to what it holds
    const size_t per = (size_t) a.T * kNh * (kHd + 2) * sizeof(float);
    while (sp.splits > 1 && per * sp.splits > a.workspace_bytes) {
        --sp.splits;
        sp.slice = (keys + sp.splits - 1) / sp.splits;
    }
    // rows before the last have fewer keys (full attention): a slice starting past a row's keys writes an empty partial
    const dim3 grid((unsigned) (a.T * kNkv), (unsigned) sp.splits);
    attn_split_kernel<<<grid, kAttnWarps * 32, 0, s>>>(a, sp.splits, sp.slice, (float*) a.workspace);
    attn_combine_kernel<<<blocks_for((int64_t) a.T * kNh * 32), 256, 0, s>>>(a, sp.splits, (const float*) a.workspace);
}

void moe_route(const float* logits, const float* bias, int T, int n_expert, int k, float w_scale, int32_t* sel, float* weight,
               void* stream) {
    moe_route_kernel<<<blocks_for((int64_t) T * 32), 256, 0, (cudaStream_t) stream>>>(logits, bias, T, n_expert, k, w_scale, sel,
                                                                                        weight);
}

void moe_plan(const int32_t* sel, int T, int k, int n_expert, int32_t* bounds, int32_t* row_of, int32_t* src_token,
              int32_t* expert_of, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    const int R = T * k;
    // the counts land in bounds[0..E), then the scan turns them into bounds in place
    moe_count_kernel<<<blocks_for((int64_t) n_expert * 32), 256, 0, s>>>(sel, R, n_expert, bounds);
    moe_scan_kernel<<<1, 32, 0, s>>>(bounds, n_expert, bounds);
    moe_place_kernel<<<blocks_for((int64_t) n_expert * 32), 256, 0, s>>>(sel, R, k, n_expert, bounds, row_of, src_token, expert_of);
}

void moe_situ(const float* gu, const int32_t* expert_of, const float* gu_scale, float* h, int R, int F, float beta, float lbeta,
              void* stream) {
    moe_situ_kernel<<<blocks_for((int64_t) R * F), 256, 0, (cudaStream_t) stream>>>(gu, expert_of, gu_scale, h, R, F, beta, lbeta);
}

void moe_combine(const float* down, const int32_t* row_of, const int32_t* sel, const float* weight, const float* d_scale,
                 float* out, int T, int k, int D, void* stream) {
    moe_combine_kernel<<<blocks_for((int64_t) T * D), 256, 0, (cudaStream_t) stream>>>(down, row_of, sel, weight, d_scale, out, T,
                                                                                         k, D);
}

void log_tau(int64_t pos0, int T, float floor, float alpha, float* tau, void* stream) {
    log_tau_kernel<<<blocks_for(T), 256, 0, (cudaStream_t) stream>>>(pos0, T, floor, alpha, tau);
}
void scale_heads(float* x, const float* tau, int T, int per_row, void* stream) {
    scale_heads_kernel<<<blocks_for((int64_t) T * per_row), 256, 0, (cudaStream_t) stream>>>(x, tau, T, per_row);
}
void sigmoid_gate(float* o, const float* g, int64_t n, void* stream) {
    sigmoid_gate_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(o, g, n);
}
void softcap(float* x, int64_t n, float cap, void* stream) {
    if (cap > 0.f) softcap_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, n, cap);
}
void kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int64_t pos0, int T, int64_t ring, void* stream) {
    kv_store_kernel<<<blocks_for((int64_t) T * kNkv * kHd), 256, 0, (cudaStream_t) stream>>>(k, v, kc, vc, pos0, T, ring);
}
void add_rows(float* x, const float* y, int64_t n, void* stream) {
    add_rows_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
}

}  // namespace halo::models::inkling2::k
