// src/models/inkling2/kernels/ink_ops.cu - the non-GEMM kernels of Inkling 2 Mini on sm_89. See ink_ops.hpp.
// clip_rows, rms_rows, loop_control and short_conv started from a local-model draft (reviewed); the attention, the
// depth mix, the MoE routing and the small ones are written for first light: correct first, then fast.
#include "models/inkling2/kernels/ink_ops.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// ---- depth_mix: D split in chunks of 256 over the card; partial sums, then the mix ----
struct StatePtrs {
    const float* p[16];
};
constexpr int kMixChunk = 256;
// part[t][0..16): sums of squares; part[t][16 + qi * 16 + j]: dot(query qi, raw state j)
__global__ void depth_mix_partial(StatePtrs st, int n, const float* __restrict__ query, int nq, float* part, int D) {
    const int64_t t = blockIdx.x;
    const int c = blockIdx.y * kMixChunk + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const float q0 = c < D ? query[c] : 0.f, q1 = (nq > 1 && c < D) ? query[D + c] : 0.f, q2 = (nq > 2 && c < D) ? query[2 * D + c] : 0.f;
    float* pt = part + t * 64;
    for (int j = 0; j < n; ++j) {
        const float v = c < D ? st.p[j][t * D + c] : 0.f;
        const float ss = warp_sum(v * v), d0 = warp_sum(v * q0);
        const float d1 = nq > 1 ? warp_sum(v * q1) : 0.f, d2 = nq > 2 ? warp_sum(v * q2) : 0.f;
        if (lane == 0) {
            atomicAdd(pt + j, ss);
            atomicAdd(pt + 16 + j, d0);
            if (nq > 1) atomicAdd(pt + 32 + j, d1);
            if (nq > 2) atomicAdd(pt + 48 + j, d2);
        }
    }
}
__global__ void depth_mix_out(StatePtrs st, int n, int nq, const float* __restrict__ part, float* __restrict__ out, int D) {
    __shared__ float inv[16], prob[3][16];
    const int64_t t = blockIdx.x;
    const float* pt = part + t * 64;
    if (threadIdx.x < 32) {
        const int j = threadIdx.x;
        const float iv = j < n ? rsqrtf(pt[j] / (float) D + 1e-6f) : 0.f;
        if (j < 16) inv[j] = iv;
        const float sc = rsqrtf((float) D);
        for (int qi = 0; qi < nq; ++qi) {
            float a = j < n ? pt[16 + qi * 16 + j] * iv * sc : -INFINITY;
            float m = a;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float e = j < n ? __expf(a - m) : 0.f;
            const float l = warp_sum(e);
            if (j < 16) prob[qi][j] = e / l;
        }
    }
    __syncthreads();
    const int c = blockIdx.y * kMixChunk + threadIdx.x;
    if (c >= D) return;
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

// ---- loop_control: norms per row, z by a warp per (row, rank), the merge by a warp per (row, value) ----
__global__ void lc_norms(const float* __restrict__ cur, const float* __restrict__ msg, float* norms, int D) {
    __shared__ float red[32];
    const int64_t t = blockIdx.x;
    float sc = 0.f, sm = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float a = cur[t * D + i], b = msg[t * D + i];
        sc += a * a, sm += b * b;
    }
    const float tc = block_sum(sc, red);
    const float tm = block_sum(sm, red);
    if (threadIdx.x == 0) norms[t * 2] = rsqrtf(tc / (float) D + 1e-6f), norms[t * 2 + 1] = rsqrtf(tm / (float) D + 1e-6f);
}
__global__ void lc_down(const float* __restrict__ cur, const float* __restrict__ msg, const float* __restrict__ norms,
                        const float* __restrict__ w_down, float* z, int T, int D, int rank) {
    const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t) T * rank) return;
    const int64_t t = wid / rank;
    const int r = (int) (wid % rank);
    const float* wr = w_down + (int64_t) r * 2 * D;
    const float* cr = cur + t * D;
    const float* mr = msg + t * D;
    float a = 0.f, b = 0.f;
    for (int i = lane; i < D; i += 32) a += wr[i] * cr[i], b += wr[D + i] * mr[i];
    a = warp_sum(a), b = warp_sum(b);
    if (lane == 0) z[wid] = a * norms[t * 2] + b * norms[t * 2 + 1];
}
__global__ void lc_merge(const float* cur, const float* __restrict__ msg, const float* __restrict__ z,
                         const float* __restrict__ w_gain, const float* __restrict__ w_forget, float* out, int T, int D, int rank) {
    const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t) T * D) return;
    const int64_t t = wid / D, i = wid % D;
    const float* zr = z + t * rank;
    float g = 0.f, f = 0.f;
    for (int r = lane; r < rank; r += 32) g += zr[r] * w_gain[i * rank + r], f += zr[r] * w_forget[i * rank + r];
    g = tanhf(warp_sum(g));
    f = tanhf(warp_sum(f));
    if (lane == 0) out[wid] = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, f), cur[wid]), __fmul_rn(g, msg[wid]));
}

// a growing device scratch per use (one session, one stream: a grow waits for the device in cudaFree)
float* scratch(int which, size_t floats) {
    static float* buf[2] = {nullptr, nullptr};
    static size_t cap[2] = {0, 0};
    if (floats > cap[which]) {
        if (buf[which]) cudaFree(buf[which]);
        cudaMalloc((void**) &buf[which], floats * sizeof(float));
        cap[which] = floats;
    }
    return buf[which];
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
    const int64_t pos0 = a.pos_dev ? *a.pos_dev : a.pos0;
    if (a.pos_dev) {   // the slice from the last row's keys
        const int64_t last = pos0 + a.T;
        const int64_t keys = a.window > 0 ? (a.window < last ? a.window : last) : last;
        slice = (keys + splits - 1) / splits;
    }
    const int64_t pt = pos0 + t;
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


// ---- decode experts: a warp per output row, the GGUF blocks dequantized in registers, FP32 activations ----

__device__ __forceinline__ float ue4m3_half(uint32_t x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) return 0.f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.f + (float) m / 8.f, e - 7);
    return raw * 0.5f;
}
__device__ __forceinline__ float half_bits(uint16_t h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }

// E2M1 doubled (the GGUF table): magnitudes 0 1 2 3 4 6 8 12, bit 3 the sign
__device__ __forceinline__ float fp4x2(uint32_t n) {
    const uint32_t m = n & 7;
    const float v = m < 4 ? (float) m : m < 6 ? (float) ((m - 2) * 2) : (float) ((m - 4) * 4);
    return (n & 8) ? -v : v;
}

// NVFP4 rows of NB blocks (64 values, 36 bytes: 4 UE4M3 scales, 32 bytes of codes): per step the warp takes 4 blocks,
// lane L the code word L % 8 of block step * 4 + L / 8 - 8 values at fixed positions, so the lane's x values load
// once (xr) and every row streams one 4-byte word a lane a step.
template <int NB>
struct Fp4Lane {
    static constexpr int kSteps = (NB + 3) / 4;
    float xr[kSteps][8];
    __device__ void load(const float* __restrict__ x, int lane) {
#pragma unroll
        for (int it = 0; it < kSteps; ++it) {
            const int b = it * 4 + lane / 8, sub = lane % 8, sb = sub >> 1, j0 = 4 * (sub & 1);
            const int base = b * 64 + 16 * sb + j0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                xr[it][i] = b < NB ? x[base + i] : 0.f;
                xr[it][4 + i] = b < NB ? x[base + 8 + i] : 0.f;
            }
        }
    }
    __device__ float dot(const uint8_t* __restrict__ row, int lane) const {
        float acc = 0.f;
        const int sub = lane % 8, sb = sub >> 1;
#pragma unroll
        for (int it = 0; it < kSteps; ++it) {
            const int b = it * 4 + lane / 8;
            if (b < NB) {
                const uint8_t* blk = row + (size_t) b * 36;
                const uint32_t scw = *reinterpret_cast<const uint32_t*>(blk);
                const uint32_t q = *reinterpret_cast<const uint32_t*>(blk + 4 + 4 * sub);
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t byte = (q >> (8 * i)) & 0xFF;
                    a += fp4x2(byte & 15) * xr[it][i] + fp4x2(byte >> 4) * xr[it][4 + i];
                }
                acc += ue4m3_half((scw >> (8 * sb)) & 0xFF) * a;
            }
        }
        return warp_sum(acc);
    }
};
// Q8_0 rows of n values: lane L takes value L of every block (34 bytes: fp16 d, 32 int8)
__device__ float q8_row_dot(const uint8_t* __restrict__ row, const float* __restrict__ x, int n, int lane) {
    float acc = 0.f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = half_bits(*reinterpret_cast<const uint16_t*>(blk));
        acc += d * (float) (int8_t) blk[2 + lane] * x[b * 32 + lane];
    }
    return warp_sum(acc);
}
__host__ __device__ inline size_t row_bytes_of(int type, int n) { return type == 40 ? (size_t) n / 64 * 36 : (size_t) n / 32 * 34; }

constexpr int kDecWarps = 8;
// one block per (pick, 8 output pairs): a warp per gate/up row pair
template <int TYPE, int NB>
__global__ void moe_dec_gu_kernel(const uint8_t* __restrict__ gu, size_t gu_bytes, const int32_t* __restrict__ sel,
                                  const float* __restrict__ x, const float* __restrict__ gu_scale, float* __restrict__ h,
                                  int k, int D, int F, float beta, float lbeta) {
    const int pick = blockIdx.x, t = pick / k;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c = blockIdx.y * kDecWarps + warp;
    if (c >= F) return;
    const int e = sel[pick];
    const uint8_t* m = gu + (size_t) e * gu_bytes;
    const size_t rb = row_bytes_of(TYPE, D);
    const float* xt = x + (size_t) t * D;
    float g0, u0;
    if (TYPE == 40) {
        Fp4Lane<NB> L;
        L.load(xt, lane);
        g0 = L.dot(m + (size_t) c * rb, lane);
        u0 = L.dot(m + (size_t) (c + F) * rb, lane);
    } else {
        g0 = q8_row_dot(m + (size_t) c * rb, xt, D, lane);
        u0 = q8_row_dot(m + (size_t) (c + F) * rb, xt, D, lane);
    }
    if (lane == 0) {
        const float sc = gu_scale ? gu_scale[e] : 1.f;
        const float g = g0 * sc, u = u0 * sc;
        const float act = beta * tanhf(g / beta) * (1.f / (1.f + __expf(-g)));
        h[(size_t) pick * F + c] = act * (lbeta > 0.f ? lbeta * tanhf(u / lbeta) : u);
    }
}
// one block per (token, 8 output rows): a warp per row over the k picks
template <int TYPE, int NB>
__global__ void moe_dec_down_kernel(const uint8_t* __restrict__ down, size_t d_bytes, const int32_t* __restrict__ sel,
                                    const float* __restrict__ weight, const float* __restrict__ d_scale,
                                    const float* __restrict__ h, float* __restrict__ out, int k, int D, int F) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.y * kDecWarps + warp;
    if (r >= D) return;
    const size_t rb = row_bytes_of(TYPE, F);
    float acc = 0.f;
    for (int s = 0; s < k; ++s) {
        const int e = sel[t * k + s];
        const float* hs = h + ((size_t) t * k + s) * F;
        const uint8_t* row = down + (size_t) e * d_bytes + (size_t) r * rb;
        float d;
        if (TYPE == 40) {
            Fp4Lane<NB> L;
            L.load(hs, lane);
            d = L.dot(row, lane);
        } else {
            d = q8_row_dot(row, hs, F, lane);
        }
        acc += weight[t * k + s] * (d_scale ? d_scale[e] : 1.f) * d;
    }
    if (lane == 0) out[(size_t) t * D + r] = acc;
}

// ---- small ----
__global__ void log_tau_kernel(int64_t pos0, const int64_t* pos_dev, int T, float floor, float alpha, float* tau) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos_dev) pos0 = *pos_dev;
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
                                int64_t pos0, const int64_t* pos_dev, int T, int64_t ring) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (pos_dev) pos0 = *pos_dev;
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
    cudaStream_t s = (cudaStream_t) stream;
    StatePtrs p{};
    const int n = std::min(n_states, 16);
    for (int j = 0; j < n; ++j) p.p[j] = states[j];
    float* part = scratch(0, (size_t) T * 64);
    cudaMemsetAsync(part, 0, (size_t) T * 64 * sizeof(float), s);
    const dim3 grid((unsigned) T, (unsigned) ((D + kMixChunk - 1) / kMixChunk));
    depth_mix_partial<<<grid, kMixChunk, 0, s>>>(p, n, query, n_query, part, D);
    depth_mix_out<<<grid, kMixChunk, 0, s>>>(p, n, n_query, part, out, D);
}

void loop_control(const float* cur, const float* msg, const float* w_down, const float* w_gain, const float* w_forget,
                  float* out, int T, int D, int rank, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    float* norms = scratch(1, (size_t) T * (2 + rank));
    float* z = norms + (size_t) T * 2;
    lc_norms<<<(unsigned) T, 256, 0, s>>>(cur, msg, norms, D);
    lc_down<<<blocks_for((int64_t) T * rank * 32), 256, 0, s>>>(cur, msg, norms, w_down, z, T, D, rank);
    lc_merge<<<blocks_for((int64_t) T * D * 32), 256, 0, s>>>(cur, msg, z, w_gain, w_forget, out, T, D, rank);
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
    const int64_t last = a.pos_dev ? a.max_keys : a.pos0 + a.T;   // keys of the last row (with pos_dev: the most)
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


void moe_dec_gu(const void* gu, size_t gu_bytes, int type, const int32_t* sel, const float* x, const float* gu_scale, float* h,
                int T, int k, int D, int F, float beta, float lbeta, void* stream) {
    const dim3 grid((unsigned) (T * k), (unsigned) ((F + kDecWarps - 1) / kDecWarps));
    cudaStream_t s = (cudaStream_t) stream;
    if (type == 40 && D == 2304) moe_dec_gu_kernel<40, 36><<<grid, kDecWarps * 32, 0, s>>>((const uint8_t*) gu, gu_bytes, sel, x, gu_scale, h, k, D, F, beta, lbeta);
    else moe_dec_gu_kernel<8, 1><<<grid, kDecWarps * 32, 0, s>>>((const uint8_t*) gu, gu_bytes, sel, x, gu_scale, h, k, D, F, beta, lbeta);
}
void moe_dec_down(const void* down, size_t d_bytes, int type, const int32_t* sel, const float* weight, const float* d_scale,
                  const float* h, float* out, int T, int k, int D, int F, void* stream) {
    const dim3 grid((unsigned) T, (unsigned) ((D + kDecWarps - 1) / kDecWarps));
    cudaStream_t s = (cudaStream_t) stream;
    if (type == 40 && F == 1152) moe_dec_down_kernel<40, 18><<<grid, kDecWarps * 32, 0, s>>>((const uint8_t*) down, d_bytes, sel, weight, d_scale, h, out, k, D, F);
    else moe_dec_down_kernel<8, 1><<<grid, kDecWarps * 32, 0, s>>>((const uint8_t*) down, d_bytes, sel, weight, d_scale, h, out, k, D, F);
}

void log_tau(int64_t pos0, const int64_t* pos_dev, int T, float floor, float alpha, float* tau, void* stream) {
    log_tau_kernel<<<blocks_for(T), 256, 0, (cudaStream_t) stream>>>(pos0, pos_dev, T, floor, alpha, tau);
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
void kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int64_t pos0, const int64_t* pos_dev, int T,
              int64_t ring, void* stream) {
    kv_store_kernel<<<blocks_for((int64_t) T * kNkv * kHd), 256, 0, (cudaStream_t) stream>>>(k, v, kc, vc, pos0, pos_dev, T, ring);
}
void reserve(int T) {
    scratch(0, (size_t) T * 64);
    scratch(1, (size_t) T * (2 + 64));
}
void add_rows(float* x, const float* y, int64_t n, void* stream) {
    add_rows_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
}

}  // namespace halo::models::inkling2::k
