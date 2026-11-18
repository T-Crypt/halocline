// src/kernels/cuda/gqa_attn.cu - see include/halo/kernels/gqa_attn.hpp.
#include "halo/kernels/gqa_attn.hpp"
#include "halo/kernels/kv_q4_layout.hpp"
#include "halo/kernels/pdl.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include "halo/log.hpp"

namespace halo::kernels {
namespace {

constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = kGqaCellChunk;  // cells per decode chunk
constexpr int kDecSplits = 64;        // a decode row's blocks per KV head (its chunks dealt round robin)
constexpr int HD = 256;              // the prompt kernel's head width
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;   // 1/sqrt(256) * log2(e)
constexpr float kNeg = -1e30f;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { halo::log::error("halo", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- decode
// Row t of a window sees positions [lo_t, hi_t): hi_t = pos0 + t + 1 (causal) or `hi`, lo_t = pos0 + t - window + 1 (a
// sliding window, else 0); position p sits in cache cell p % cap.  Cells are split in CHUNK-position chunks from chunk
// `base`; block b of a row takes chunks b, b + gx, ... (gx from the capacity alone) and folds them in order, so a row's
// sums depend on its own range only, never on the window's rows or where the first cell is read from.
struct Range {
    int64_t pos0 = 0, hi = 0, window = 0, cap = 0;
    int causal = 1, base = 0;
    const int32_t* pos_dev = nullptr;   // pos0 read from the device (a captured window)
};
__device__ __forceinline__ int64_t row_pos0(const Range& r) { return r.pos_dev ? (int64_t) *r.pos_dev : r.pos0; }
// A split cache (KvFar): true when cell c lives in the far buffers, `c` then its row there
__device__ __forceinline__ bool kv_far(const KvFar& f, int64_t& c) {
    if (c < f.at) return false;
    c -= f.at;
    return true;
}

// The windows one launch takes: segment k is rows [row0[k], row0[k + 1]) of q, the partials and the output, its 16-row
// tiles of (row, head) pairs [tile0[k], tile0[k + 1]) of the grid, with its own cache and range.  Each block and each
// merged row computes exactly what a launch of its segment alone would.
constexpr int kMaxSegs = kGqaMaxWindow;
struct DecSegs {
    int n = 1;
    int row0[kMaxSegs + 1] = {};
    int tile0[kMaxSegs + 1] = {};
    const __half* kc[kMaxSegs] = {};
    const __half* vc[kMaxSegs] = {};
    KvInt8 q8[kMaxSegs] = {};
    KvK8V4 k8v4[kMaxSegs] = {};
    KvFar far_kv[kMaxSegs] = {};
    Range r[kMaxSegs];
    KvQ4 q4[kMaxSegs] = {};   // last: the fields before it keep the offsets the K8V4 / int8 / fp16 kernels read
};
__device__ __forceinline__ int seg_of(const int* first, int n, int i) {
    int k = 0;
    while (k + 1 < n && i >= first[k + 1]) ++k;
    return k;
}
__device__ __forceinline__ int64_t row_hi(const Range& r, int t) { return r.causal ? row_pos0(r) + t + 1 : r.hi; }
__device__ __forceinline__ int64_t row_lo(const Range& r, int t) {
    return r.window > 0 ? max((int64_t) 0, row_pos0(r) + t - r.window + 1) : 0;
}

// Block (query head, row t), one thread per dim: the row's splits merged in order.  Splits past the row's last chunk
// saw no cell of it ({max kNeg, sum 0, zeros}: every term they add is an exact zero), so the merge stops at the last
// split that holds one of its cells; the sums and their order are those of the full loop.
template <int HD>
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* part, int n_head, int gx, float* out, const float* gate_q,
                                                          uint16_t* gated16, const DecSegs segs) {
    pdl_launch_dependents();
    pdl_wait();
    const int h = blockIdx.x, row = blockIdx.y, d = threadIdx.x;   // row: of the launch, t: of its segment
    const int k = seg_of(segs.row0, segs.n, row), t = row - segs.row0[k];
    const Range& r = segs.r[k];
    const float* p = part + ((int64_t) row * n_head + h) * gx * (HD + 2);
    const int64_t chunks = (row_hi(r, t) + CHUNK - 1) / CHUNK - r.base;   // the row's chunks from the first split's
    const int used = (int) max((int64_t) 1, min((int64_t) gx, chunks));
    float m = kNeg;
    for (int b = 0; b < used; ++b) m = fmaxf(m, p[(int64_t) b * (HD + 2)]);
    float l = 0.0f, a = 0.0f;
    for (int b = 0; b < used; ++b) {
        const float* pb = p + (int64_t) b * (HD + 2);
        const float w = exp2f(pb[0] - m);
        l = fmaf(pb[1], w, l);
        a = fmaf(pb[2 + d], w, a);
    }
    const float o = a / l;
    const int64_t i = ((int64_t) row * n_head + h) * HD + d;
    if (out) out[i] = o;
    if (gated16) {
        const float g = gate_q[((int64_t) row * n_head + h) * 2 * HD + HD + d];
        gated16[i] = __bfloat16_as_ushort(__float2bfloat16_rn(o / (1.0f + expf(-g))));
    }
}

// the splits a row's chunks go to: from the capacity's chunks alone
int decode_splits(int64_t n_chunks) { return (int) std::max<int64_t>(1, std::min<int64_t>(kDecSplits, n_chunks)); }

// ---------------------------------------------------------------- prompt: shared helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816_h(unsigned (&c)[2], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                           unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16832_s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}

// global -> shared without a register or a wait, or zeros when !ok (the src-size form zero-fills the rest of the row):
// the K8V4 decode kernel stages a chunk's raw bytes a chunk ahead of the dequant that reads them.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---------------------------------------------------------------- decode on the tensor cores
// A decode chunk's cells [from, to) of K and V as fp16 in `ks` / `vs` (int8 and K8V4 caches: code x scale), zero past
// the rows' last cell.  Both decode kernels load a chunk through this, so the bytes a row reads and their rounding are
// one function.  Q4 (rotated q4_0 K and V, --kv q4_0) is its own instantiation, never a runtime branch, so the
// instantiations the other formats run are the code they were before the 4-bit cache existed.
template <int HD, bool Q4 = false>
__device__ __forceinline__ void dec_load_kv(__half (*ks)[HD + 8], __half (*vs)[HD + 8], const Range& r, int kvh, int n_kv,
                                            const __half* kc, const __half* vc, KvInt8 q8, KvK8V4 k8v4, const KvFar& fr,
                                            int64_t c0, int64_t hi_max, int from, int to, int tid, KvQ4 q4 = {}) {
    const int64_t cell0 = c0 % r.cap;
    for (int idx = tid; idx < CHUNK * HD / 8; idx += 128) {
        const int cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
        if (cell < from || cell >= to) continue;
        uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
        if (c0 + cell < hi_max) {
            int64_t ce = cell0 + cell;
            if (ce >= r.cap) ce -= r.cap;
            const bool far_kv = kv_far(fr, ce);
            const int64_t row = ce * n_kv + kvh;
            if constexpr (Q4) {
                // K and V both rotated q4_0 rows: the eight dims are the block that covers them, with the block's own
                // scale (the caller rotated the queries by the same transform); a 4-bit cache has no far tier
                (void) far_kv;
                const int64_t off = row * (int64_t) kv_q4_head_bytes(HD);
                const KvQ4Block* kb = reinterpret_cast<const KvQ4Block*>(q4.k + off) + (d8 >> 5);
                const KvQ4Block* vb = reinterpret_cast<const KvQ4Block*>(q4.v + off) + (d8 >> 5);
                const float kd = f32_from_f16(kb->d), vd = f32_from_f16(vb->d);
                int kq[8], vq[8];
                kv_q4_codes8(kb->qs, d8, kq);
                kv_q4_codes8(vb->qs, d8, vq);
                __half2 kh[4], vh[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    kh[e] = __floats2half2_rn(kv_q4_dequant(kq[2 * e], kd), kv_q4_dequant(kq[2 * e + 1], kd));
                    vh[e] = __floats2half2_rn(kv_q4_dequant(vq[2 * e], vd), kv_q4_dequant(vq[2 * e + 1], vd));
                }
                kw = *reinterpret_cast<uint4*>(kh);
                vw = *reinterpret_cast<uint4*>(vh);
            } else if (q8.k) {
                const KvInt8& c8 = far_kv ? fr.q8 : q8;
                const float sk = __half2float(__ushort_as_half(c8.ks[row * (HD / 64) + d8 / 64]));
                const float sv = __half2float(__ushort_as_half(c8.vs[row * (HD / 64) + d8 / 64]));
                int8_t kb[8], vb[8];
                *reinterpret_cast<int2*>(kb) = *reinterpret_cast<const int2*>(c8.k + row * HD + d8);
                *reinterpret_cast<int2*>(vb) = *reinterpret_cast<const int2*>(c8.v + row * HD + d8);
                __half2 kh[4], vh[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    kh[e] = __floats2half2_rn((float) kb[2 * e] * sk, (float) kb[2 * e + 1] * sk);
                    vh[e] = __floats2half2_rn((float) vb[2 * e] * sv, (float) vb[2 * e + 1] * sv);
                }
                kw = *reinterpret_cast<uint4*>(kh);
                vw = *reinterpret_cast<uint4*>(vh);
            } else if (k8v4.k) {
                // K8V4: K as the int8 cache holds it, V as the rotated q4_0 block that covers these 8 dims
                const float sk = __half2float(__ushort_as_half(k8v4.ks[row * (HD / 64) + d8 / 64]));
                int8_t kb[8];
                *reinterpret_cast<int2*>(kb) = *reinterpret_cast<const int2*>(k8v4.k + row * HD + d8);
                __half2 kh[4], vh[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    kh[e] = __floats2half2_rn((float) kb[2 * e] * sk, (float) kb[2 * e + 1] * sk);
                kw = *reinterpret_cast<uint4*>(kh);
                const KvQ4Block* blk =
                    reinterpret_cast<const KvQ4Block*>(k8v4.vq + row * (int64_t) kv_q4_head_bytes(HD)) + (d8 >> 5);
                const float vd = f32_from_f16(blk->d);
                int codes[8];
                kv_q4_codes8(blk->qs, d8, codes);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    vh[e] = __floats2half2_rn(kv_q4_dequant(codes[2 * e], vd), kv_q4_dequant(codes[2 * e + 1], vd));
                vw = *reinterpret_cast<uint4*>(vh);
            } else {
                kw = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.k : kc) + row * HD + d8);
                vw = *reinterpret_cast<const uint4*>((far_kv ? (const __half*) fr.v : vc) + row * HD + d8);
            }
        }
        *reinterpret_cast<uint4*>(&ks[cell][d8]) = kw;
        *reinterpret_cast<uint4*>(&vs[cell][d8]) = vw;
    }
}

// Block (split b, KV head, 16 query rows): the rows are the window's (row, head) pairs of the KV group, m = t G + h.  Its
// chunks b, b + gx, ... of CHUNK cells: K and V decoded to fp16 in shared memory, S = Q K^T on mma m16n8k16 (fp16 in,
// fp32 accumulate), an online softmax per row over its cells, O = O alpha + P V (P in fp16) on mma; the rows' partials
// {max, sum, O} go to the merge.  Every row's arithmetic is its own (mma columns and rows never meet), the chunks follow
// positions and the splits the capacity, so a row's bits are the same at any window width.
template <int HD, bool Q4 = false>
// (launched through launch_pdl: q and the cache come from the kernels before, read after pdl_wait, so not __restrict__)
__global__ void __launch_bounds__(128) decode_mma_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs, int n_chunks,
                                                         float scale, float* part_all) {
    constexpr int LD = HD + 8;          // padded rows (halves): ldmatrix without bank conflicts
    constexpr int NT = HD / 32;         // P V: a warp's 8-dim tiles
    __shared__ __align__(16) __half qs[16][LD];
    __shared__ __align__(16) __half ks[CHUNK][LD];
    __shared__ __align__(16) __half vs[CHUNK][LD];
    __shared__ float ss[16][CHUNK + 1];
    __shared__ __align__(16) __half ps[16][CHUNK + 8];
    __shared__ float rm[16], rl[16], ra[16];
    __shared__ int64_t rlo[16], rhi[16];
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = seg_of(segs.tile0, segs.n, (int) blockIdx.z), row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvK8V4 k8v4 = segs.k8v4[sk];
    const KvQ4 q4 = segs.q4[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G, m0 = ((int) blockIdx.z - segs.tile0[sk]) * 16;
    // before the wait: the rows' ranges (positions come from the host) and, of the block's first chunk, the cells below
    // the window's first position, which earlier windows wrote; the window's own cells and the queries come after it
    if (tid < 16) {
        const int m = m0 + tid;
        rlo[tid] = m < M ? row_lo(r, m / G) : 0;
        rhi[tid] = m < M ? row_hi(r, m / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[i]);
    const int64_t first_c0 = (int64_t) (r.base + blockIdx.x) * CHUNK;
    const int64_t own = row_pos0(r);   // cells from the window's first position on are this window's (written just before)
    const int pre = (int) max((int64_t) 0, min((int64_t) CHUNK, own - first_c0));
    if ((int) blockIdx.x < n_chunks && first_c0 < hi_max && pre > 0)
        dec_load_kv<HD, Q4>(ks, vs, r, kvh, n_kv, kc, vc, q8, k8v4, fr, first_c0, hi_max, 0, pre, tid, q4);
    pdl_wait();
    for (int idx = tid; idx < 16 * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        const float v = m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi_max) break;   // past every row's cells
        // the first chunk's earlier cells are already in
        dec_load_kv<HD, Q4>(ks, vs, r, kvh, n_kv, kc, vc, q8, k8v4, fr, c0, hi_max,
                            chunk == (int) blockIdx.x ? pre : 0, CHUNK, tid, q4);
        __syncthreads();
        // S = Q K^T: warp w the cells 8w .. 8w + 7
        for (int nc = warp; nc < CHUNK / 8; nc += 4) {
            float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < HD; k += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, &qs[lane & 15][k + (lane >> 4) * 8]);
                ldsm_x2(b0, b1, &ks[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s4, a0, a1, a2, a3, b0, b1);
            }
            ss[g][nc * 8 + 2 * c] = s4[0];
            ss[g][nc * 8 + 2 * c + 1] = s4[1];
            ss[g + 8][nc * 8 + 2 * c] = s4[2];
            ss[g + 8][nc * 8 + 2 * c + 1] = s4[3];
        }
        __syncthreads();
        // each row: the chunk's max over its cells, p = exp2(s - max') there (0 elsewhere), the running {max, sum}
        for (int i = warp; i < 16; i += 4) {
            float pv[CHUNK / 32], mx = kNeg;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const int cc = lane + 32 * u;
                const int64_t pos = c0 + cc;
                pv[u] = pos >= rlo[i] && pos < rhi[i] ? ss[i][cc] : -INFINITY;
                mx = fmaxf(mx, pv[u]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mold = rm[i], mnew = fmaxf(mold, mx);
            float l = 0.0f;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const float p = pv[u] == -INFINITY ? 0.0f : exp2f(pv[u] - mnew);
                ps[i][lane + 32 * u] = __float2half_rn(p);
                l += p;
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float alpha = exp2f(mold - mnew);
                ra[i] = alpha;
                rl[i] = fmaf(rl[i], alpha, l);
                rm[i] = mnew;
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w the dims w HD / 4 ..
        {
            const float ag = ra[g], ag8 = ra[g + 8];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                acc[n][0] *= ag;
                acc[n][1] *= ag;
                acc[n][2] *= ag8;
                acc[n][3] *= ag8;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &ps[lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, &vs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
                    mma16816(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();   // the shared tiles are free for the next chunk
    }
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int i = g + 8 * hr, m = m0 + i;
        if (m >= M) continue;
        float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
        if (warp == 0 && c == 0) {
            out[0] = rm[i];
            out[1] = rl[i];
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int d = warp * (HD / 4) + n * 8 + 2 * c;
            out[2 + d] = acc[n][2 * hr];
            out[3 + d] = acc[n][2 * hr + 1];
        }
    }
}

// ---------------------------------------------------------------- decode: a whole window in one block
// Shared bytes of the window kernel: R groups of 16 rows' q on padded fp16 rows, the chunk's K and V tile (loaded once
// and read by every group), then each group's scores, probabilities, running {max, sum, weight} and its rows' ranges.
// __host__ __device__: the launch asks for these bytes and the kernel checks its regions against them.
template <int HD, int R>
__host__ __device__ constexpr size_t dec_win_smem() {
    constexpr size_t row = (size_t) (HD + 8) * sizeof(__half);
    return (R * 16 * row + 2 * CHUNK * row + R * 16 * (CHUNK + 1) * sizeof(float) +
            R * 16 * (CHUNK + 8) * sizeof(__half) + 3 * R * 16 * sizeof(float) + 2 * R * 16 * sizeof(int64_t) + 15) &
           ~(size_t) 15;
}

// Block (split b, KV head, segment): every row of the segment's window for this KV head, m = t G + h < M = T G, in
// ceil(M / 16) <= R 16-row m16 groups.  Its chunk's K and V are decoded into shared memory once and every group computes
// S = Q K^T and O += P V from that one copy, so a window reads its cells once per split instead of once per 16-row group.
// Group gr holds the 16 rows the tiled kernel's tile gr holds and runs its statements, so a row's bits are the same at
// any window width, any R and against that kernel.
template <int HD, int R, bool Q4 = false>
__global__ void __launch_bounds__(128) decode_mma_window_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                               int n_chunks, float scale, float* part_all) {
    constexpr int LD = HD + 8;          // padded rows (halves): ldmatrix without bank conflicts
    constexpr int NT = HD / 32;         // P V: a warp's 8-dim tiles
    constexpr int NR = R * 16;          // the block's rows of this KV head
    constexpr size_t kRow = (size_t) LD * sizeof(__half);
    extern __shared__ __align__(16) unsigned char smem[];
    __half (*qs)[LD] = reinterpret_cast<__half (*)[LD]>(smem);
    __half (*ks)[LD] = reinterpret_cast<__half (*)[LD]>(smem + NR * kRow);
    __half (*vs)[LD] = reinterpret_cast<__half (*)[LD]>(smem + (NR + CHUNK) * kRow);
    float (*ss)[CHUNK + 1] = reinterpret_cast<float (*)[CHUNK + 1]>(smem + (NR + 2 * CHUNK) * kRow);
    __half (*ps)[CHUNK + 8] = reinterpret_cast<__half (*)[CHUNK + 8]>(ss + NR);
    float* rm = reinterpret_cast<float*>(ps + NR);
    float* rl = rm + NR, *ra = rl + NR;
    int64_t* rlo = reinterpret_cast<int64_t*>(ra + NR);
    int64_t* rhi = rlo + NR;
    // the bytes the host asks for are the ones these regions span (the two are written apart)
    static_assert(dec_win_smem<HD, R>() ==
                      (NR + 2 * CHUNK) * kRow + NR * (CHUNK + 1) * sizeof(float) + NR * (CHUNK + 8) * sizeof(__half) +
                          3 * NR * sizeof(float) + 2 * NR * sizeof(int64_t),
                  "the window kernel's shared bytes cover its regions");
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = (int) blockIdx.z, row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const Range r = segs.r[sk];
    const __half* kc = segs.kc[sk];
    const __half* vc = segs.vc[sk];
    const KvInt8 q8 = segs.q8[sk];
    const KvK8V4 k8v4 = segs.k8v4[sk];
    const KvQ4 q4 = segs.q4[sk];
    const KvFar fr = segs.far_kv[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G;
    // before the wait: the rows' ranges (positions come from the host) and, of the block's first chunk, the cells below
    // the window's first position, which earlier windows wrote; the window's own cells and the queries come after it
    if (tid < NR) {
        rlo[tid] = tid < M ? row_lo(r, tid / G) : 0;
        rhi[tid] = tid < M ? row_hi(r, tid / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    __syncthreads();
    // the last cell of the window's widest row: cells past it load as zeros and the sweep ends past it
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < NR; ++i) hi_max = max(hi_max, rhi[i]);
    const int64_t first_c0 = (int64_t) (r.base + blockIdx.x) * CHUNK;
    const int64_t own = row_pos0(r);   // cells from the window's first position on are this window's (written just before)
    const int pre = (int) max((int64_t) 0, min((int64_t) CHUNK, own - first_c0));
    if ((int) blockIdx.x < n_chunks && first_c0 < hi_max && pre > 0)
        dec_load_kv<HD, Q4>(ks, vs, r, kvh, n_kv, kc, vc, q8, k8v4, fr, first_c0, hi_max, 0, pre, tid, q4);
    pdl_wait();
    for (int idx = tid; idx < NR * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD;
        const float v = i < M ? q[((int64_t) (i / G) * n_head + kvh * G + i % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    float acc[R][NT][4];
#pragma unroll
    for (int gr = 0; gr < R; ++gr)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[gr][n][0] = acc[gr][n][1] = acc[gr][n][2] = acc[gr][n][3] = 0.0f;
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi_max) break;   // past every row's cells
        // the first chunk's earlier cells are already in
        dec_load_kv<HD, Q4>(ks, vs, r, kvh, n_kv, kc, vc, q8, k8v4, fr, c0, hi_max,
                            chunk == (int) blockIdx.x ? pre : 0, CHUNK, tid, q4);
        __syncthreads();
        // S = Q K^T for every group: warp w the cells 8w .. 8w + 7
        for (int nc = warp; nc < CHUNK / 8; nc += 4) {
#pragma unroll
            for (int gr = 0; gr < R; ++gr) {
                if (16 * gr >= M) continue;   // the group holds no row of this window
                float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int k = 0; k < HD; k += 16) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldsm_x4(a0, a1, a2, a3, &qs[16 * gr + (lane & 15)][k + (lane >> 4) * 8]);
                    ldsm_x2(b0, b1, &ks[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                    mma16816(s4, a0, a1, a2, a3, b0, b1);
                }
                ss[16 * gr + g][nc * 8 + 2 * c] = s4[0];
                ss[16 * gr + g][nc * 8 + 2 * c + 1] = s4[1];
                ss[16 * gr + g + 8][nc * 8 + 2 * c] = s4[2];
                ss[16 * gr + g + 8][nc * 8 + 2 * c + 1] = s4[3];
            }
        }
        __syncthreads();
        // each row: the chunk's max over its cells, p = exp2(s - max') there (0 elsewhere), the running {max, sum}
#pragma unroll
        for (int gr = 0; gr < R; ++gr) {
            if (16 * gr >= M) continue;
            for (int i = warp; i < 16; i += 4) {
                const int row = 16 * gr + i;
                float pv[CHUNK / 32], mx = kNeg;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const int cc = lane + 32 * u;
                    const int64_t pos = c0 + cc;
                    pv[u] = pos >= rlo[row] && pos < rhi[row] ? ss[row][cc] : -INFINITY;
                    mx = fmaxf(mx, pv[u]);
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                const float mold = rm[row], mnew = fmaxf(mold, mx);
                float l = 0.0f;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const float p = pv[u] == -INFINITY ? 0.0f : exp2f(pv[u] - mnew);
                    ps[row][lane + 32 * u] = __float2half_rn(p);
                    l += p;
                }
                l = warp_sum(l);
                if (lane == 0) {
                    const float alpha = exp2f(mold - mnew);
                    ra[row] = alpha;
                    rl[row] = fmaf(rl[row], alpha, l);
                    rm[row] = mnew;
                }
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w the dims w HD / 4 ..
#pragma unroll
        for (int gr = 0; gr < R; ++gr) {
            if (16 * gr >= M) continue;
            const float ag = ra[16 * gr + g], ag8 = ra[16 * gr + g + 8];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                acc[gr][n][0] *= ag;
                acc[gr][n][1] *= ag;
                acc[gr][n][2] *= ag8;
                acc[gr][n][3] *= ag8;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &ps[16 * gr + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, &vs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
                    mma16816(acc[gr][n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();   // the shared tiles are free for the next chunk
    }
#pragma unroll
    for (int gr = 0; gr < R; ++gr) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int m = 16 * gr + g + 8 * hr;
            if (m >= M) continue;
            float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
            if (warp == 0 && c == 0) {
                out[0] = rm[m];
                out[1] = rl[m];
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int d = warp * (HD / 4) + n * 8 + 2 * c;
                out[2 + d] = acc[gr][n][2 * hr];
                out[3 + d] = acc[gr][n][2 * hr + 1];
            }
        }
    }
}

int opt_in_device();   // the current device, for a per-device shared-memory opt-in (defined below)

// the window kernel's launch: one block a (split, KV head, segment), its shared bytes opted in once per instantiation
// and device
template <int HD, int R, bool Q4 = false>
void launch_win(const float* q, int n_head, int n_kv, const DecSegs& segs, float scale, int gx, int n_chunks, float* scratch,
                cudaStream_t s) {
    const size_t smem = dec_win_smem<HD, R>();
    static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
    const int dev = opt_in_device();
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(decode_mma_window_kernel<HD, R, Q4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
            cudaSuccess) {
            halo::log::error("halo", "gqa decode attention: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    launch_pdl(decode_mma_window_kernel<HD, R, Q4>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.n), dim3(128), smem, s, q,
               n_head, n_kv, segs, n_chunks, scale, scratch);
}

// ---------------------------------------------------------------- decode: K8V4, a window in one block, raw bytes staged
// The served cache's decode kernel, for a window of M = T G <= 48 rows: every row of the window for one KV head in one
// block, in R = ceil(M / 16) 16-row m16 groups, W warps (8 at R <= 2, 12 at R = 3).  Its chunk's raw compressed bytes
// (K's codes and scales and V's rotated q4 blocks, 13,056 B) go to shared memory with cp.async a chunk ahead; the dequant
// reads them there and writes the same fp16 K and V tile the other decode kernels build, with the same formulas.  The
// warps then work in parallel: a group's n-tiles for S = Q K^T, a row each for the online softmax, a dim slice of its
// group's O for P V.  A row runs the tiled kernel's statements inside the tiled kernel's 16-row group, so its values are
// that kernel's; only which warp runs a statement changes.

// Shared bytes of the K8V4 decode kernel: R groups of 16 rows' q on padded fp16 rows, the chunk's K and V as fp16, each
// group's scores, probabilities, running {max, sum, weight} and its rows' ranges, then the chunk's raw bytes.
template <int HD, int R>
__host__ __device__ constexpr size_t dec_k8_smem() {
    constexpr size_t row = (size_t) (HD + 8) * sizeof(__half);
    constexpr size_t raw = (size_t) CHUNK * (HD + HD / 64 * 2 + HD / kKvQ4Block * (int) sizeof(KvQ4Block));
    return (R * 16 * row + 2 * CHUNK * row + R * 16 * (CHUNK + 1) * sizeof(float) +
            R * 16 * (CHUNK + 8) * sizeof(__half) + 3 * R * 16 * sizeof(float) + 2 * R * 16 * sizeof(int64_t) + raw + 15) &
           ~(size_t) 15;
}

// One chunk's raw bytes ([from, to) of the chunk whose first cell is c0) from the K8V4 cache into the raw region: K's
// codes, K's per-64-dim scales, V's q4 blocks, at the offsets the dequant reads them.  Cells the chunk has no use for
// (outside [from, to), or past every row's last cell) copy nothing and read as zeros.
template <int HD, int THREADS>
__device__ __forceinline__ void dec_k8_cp(unsigned char* rawK, unsigned char* rawS, unsigned char* rawV, KvK8V4 k8v4, int kvh,
                                          int n_kv, const Range& r, int64_t c0, int64_t hi_max, int from, int to, int tid) {
    constexpr int kQ4B = HD / kKvQ4Block * (int) sizeof(KvQ4Block);
    const int64_t cell0 = c0 % r.cap;
    for (int u = tid; u < CHUNK * (HD / 16); u += THREADS) {
        const int cell = u / (HD / 16), d16 = (u - cell * (HD / 16)) * 16;
        int64_t ce = cell0 + cell;
        if (ce >= r.cap) ce -= r.cap;
        cp16(rawK + cell * HD + d16, k8v4.k + (ce * n_kv + kvh) * HD + d16,
             cell >= from && cell < to && c0 + cell < hi_max);
    }
    for (int cell = tid; cell < CHUNK; cell += THREADS) {
        int64_t ce = cell0 + cell;
        if (ce >= r.cap) ce -= r.cap;
        cp8(rawS + cell * (HD / 64) * 2, k8v4.ks + (ce * n_kv + kvh) * (HD / 64),
            cell >= from && cell < to && c0 + cell < hi_max);
    }
    for (int u = tid; u < CHUNK * (kQ4B / 16); u += THREADS) {
        const int cell = u / (kQ4B / 16), b = (u - cell * (kQ4B / 16)) * 16;
        int64_t ce = cell0 + cell;
        if (ce >= r.cap) ce -= r.cap;
        cp16(rawV + cell * kQ4B + b, k8v4.vq + (ce * n_kv + kvh) * kQ4B + b,
             cell >= from && cell < to && c0 + cell < hi_max);
    }
}

// The staged chunk's cells [from, to) of K and V as fp16 in `ks` / `vs`: code x scale for K, the rotated q4_0 block for
// V, each rounded to fp16 as dec_load_kv rounds it, and zeros past the rows' last cell (the same four statements).
template <int HD, int THREADS>
__device__ __forceinline__ void dec_k8_dequant(__half (*ks)[HD + 8], __half (*vs)[HD + 8], const unsigned char* rawK,
                                               const unsigned char* rawS, const unsigned char* rawV, int64_t c0, int64_t hi_max,
                                               int from, int to, int tid) {
    constexpr int kQ4B = HD / kKvQ4Block * (int) sizeof(KvQ4Block);
    for (int idx = tid; idx < CHUNK * HD / 8; idx += THREADS) {
        const int cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
        if (cell < from || cell >= to) continue;
        uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
        if (c0 + cell < hi_max) {
            const float sk = __half2float(
                __ushort_as_half(*reinterpret_cast<const uint16_t*>(rawS + cell * (HD / 64) * 2 + (d8 >> 6) * 2)));
            const int8_t* kb = reinterpret_cast<const int8_t*>(rawK + cell * HD + d8);
            __half2 kh[4], vh[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) kh[e] = __floats2half2_rn((float) kb[2 * e] * sk, (float) kb[2 * e + 1] * sk);
            kw = *reinterpret_cast<uint4*>(kh);
            const KvQ4Block* blk = reinterpret_cast<const KvQ4Block*>(rawV + cell * kQ4B) + (d8 >> 5);
            const float vd = f32_from_f16(blk->d);
            int codes[8];
            kv_q4_codes8(blk->qs, d8, codes);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                vh[e] = __floats2half2_rn(kv_q4_dequant(codes[2 * e], vd), kv_q4_dequant(codes[2 * e + 1], vd));
            vw = *reinterpret_cast<uint4*>(vh);
        }
        *reinterpret_cast<uint4*>(&ks[cell][d8]) = kw;
        *reinterpret_cast<uint4*>(&vs[cell][d8]) = vw;
    }
}

// Block (split b, KV head, segment): every row of the segment's window for this KV head, W warps.  Its chunks b, b + gx,
// ... of CHUNK cells each are dequantized once for all of the block's groups and folded in order, so a row's sums depend
// on its own range only and its partials are the ones the merge expects at the same addresses.
template <int HD, int R, int W>
__global__ void __launch_bounds__(W * 32) decode_k8v4_kernel(const float* q_all, int n_head, int n_kv, const DecSegs segs,
                                                             int n_chunks, float scale, float* part_all) {
    constexpr int LD = HD + 8;          // padded rows (halves): ldmatrix without bank conflicts
    constexpr int WPG = W / R;          // warps a group: its n-tiles for S, its dim slices for O
    constexpr int NT = HD / 8 / WPG;    // P V: the 8-dim tiles one warp owns
    constexpr int NR = R * 16;
    constexpr int kQ4B = HD / kKvQ4Block * (int) sizeof(KvQ4Block);
    constexpr size_t kRow = (size_t) LD * sizeof(__half);
    static_assert((HD / 8) % WPG == 0, "the decode kernel's warps split a group's output dims");
    extern __shared__ __align__(16) unsigned char smem[];
    __half (*qs)[LD] = reinterpret_cast<__half (*)[LD]>(smem);
    __half (*ks)[LD] = reinterpret_cast<__half (*)[LD]>(smem + NR * kRow);
    __half (*vs)[LD] = reinterpret_cast<__half (*)[LD]>(smem + (NR + CHUNK) * kRow);
    float (*ss)[CHUNK + 1] = reinterpret_cast<float (*)[CHUNK + 1]>(smem + (NR + 2 * CHUNK) * kRow);
    __half (*ps)[CHUNK + 8] = reinterpret_cast<__half (*)[CHUNK + 8]>(ss + NR);
    float* rm = reinterpret_cast<float*>(ps + NR);
    float* rl = rm + NR, *ra = rl + NR;
    int64_t* rlo = reinterpret_cast<int64_t*>(ra + NR);
    int64_t* rhi = rlo + NR;
    unsigned char* rawK = reinterpret_cast<unsigned char*>(rhi + NR);
    unsigned char* rawS = rawK + CHUNK * HD;
    unsigned char* rawV = rawS + CHUNK * (HD / 64) * 2;
    // the bytes the host asks for are the ones these regions span (the two are written apart)
    static_assert(dec_k8_smem<HD, R>() ==
                      (NR + 2 * CHUNK) * kRow + NR * (CHUNK + 1) * sizeof(float) + NR * (CHUNK + 8) * sizeof(__half) +
                          3 * NR * sizeof(float) + 2 * NR * sizeof(int64_t) + (size_t) CHUNK * (HD + HD / 64 * 2 + kQ4B),
                  "the K8V4 decode kernel's shared bytes cover its regions");
    pdl_launch_dependents();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv;
    const int sk = (int) blockIdx.z, row0 = segs.row0[sk], T = segs.row0[sk + 1] - row0;
    const Range r = segs.r[sk];
    const KvK8V4 k8v4 = segs.k8v4[sk];
    const float* q = q_all + (int64_t) row0 * n_head * HD;
    float* part = part_all + (int64_t) row0 * n_head * gx * (HD + 2);
    const int M = T * G;
    // before the wait: the rows' ranges (positions come from the host) and, of the block's first chunk, the cells below
    // the window's first position, which earlier windows wrote; the window's own cells and the queries come after it
    if (tid < NR) {
        rlo[tid] = tid < M ? row_lo(r, tid / G) : 0;
        rhi[tid] = tid < M ? row_hi(r, tid / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    __syncthreads();
    // the last cell of the window's widest row: cells past it load as zeros and the sweep ends past it
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < NR; ++i) hi_max = max(hi_max, rhi[i]);
    const int64_t first_c0 = (int64_t) (r.base + blockIdx.x) * CHUNK;
    const int64_t own = row_pos0(r);   // cells from the window's first position on are this window's (written just before)
    const int pre = (int) max((int64_t) 0, min((int64_t) CHUNK, own - first_c0));
    const bool live = (int) blockIdx.x < n_chunks && first_c0 < hi_max;
    if (live && pre > 0) {
        dec_k8_cp<HD, W * 32>(rawK, rawS, rawV, k8v4, kvh, n_kv, r, first_c0, hi_max, 0, pre, tid);
        cp_commit();
        cp_wait<0>();
        __syncthreads();   // the earlier cells have landed in every thread's copies
        dec_k8_dequant<HD, W * 32>(ks, vs, rawK, rawS, rawV, first_c0, hi_max, 0, pre, tid);
        __syncthreads();
    }
    pdl_wait();
    for (int idx = tid; idx < NR * HD; idx += W * 32) {
        const int i = idx / HD, d = idx - i * HD;
        const float v = i < M ? q[((int64_t) (i / G) * n_head + kvh * G + i % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    // the first chunk's own cells go in flight now; every later chunk stages its successor mid-loop, so the copy of one
    // chunk overlaps the dequant and the mma of the one before it
    if (live) dec_k8_cp<HD, W * 32>(rawK, rawS, rawV, k8v4, kvh, n_kv, r, first_c0, hi_max, pre, CHUNK, tid);
    cp_commit();
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi_max) break;   // past every row's cells
        cp_wait<0>();
        __syncthreads();           // this chunk's raw bytes are in shared, for every thread
        // the first chunk's earlier cells are already in
        dec_k8_dequant<HD, W * 32>(ks, vs, rawK, rawS, rawV, c0, hi_max,
                                   chunk == (int) blockIdx.x ? pre : 0, CHUNK, tid);
        __syncthreads();           // the fp16 tile is ready for the mma and the raw region is free
        const int64_t nc0 = (int64_t) (r.base + chunk + gx) * CHUNK;
        if (chunk + gx < n_chunks && nc0 < hi_max)
            dec_k8_cp<HD, W * 32>(rawK, rawS, rawV, k8v4, kvh, n_kv, r, nc0, hi_max, 0, CHUNK, tid);
        cp_commit();
        // S = Q K^T: warp w group w / 4's 8-cell tile w % 4 (a group's four tiles in parallel)
        for (int t = warp; t < R * (CHUNK / 8); t += W) {
            const int gr = t / (CHUNK / 8), nc = t - gr * (CHUNK / 8);
            float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < HD; k += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, &qs[16 * gr + (lane & 15)][k + (lane >> 4) * 8]);
                ldsm_x2(b0, b1, &ks[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s4, a0, a1, a2, a3, b0, b1);
            }
            ss[16 * gr + g][nc * 8 + 2 * c] = s4[0];
            ss[16 * gr + g][nc * 8 + 2 * c + 1] = s4[1];
            ss[16 * gr + g + 8][nc * 8 + 2 * c] = s4[2];
            ss[16 * gr + g + 8][nc * 8 + 2 * c + 1] = s4[3];
        }
        __syncthreads();
        // each row: the chunk's max over its cells, p = exp2(s - max') there (0 elsewhere), the running {max, sum}
        for (int row = warp; row < NR; row += W) {
            float pv[CHUNK / 32], mx = kNeg;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const int cc = lane + 32 * u;
                const int64_t pos = c0 + cc;
                pv[u] = pos >= rlo[row] && pos < rhi[row] ? ss[row][cc] : -INFINITY;
                mx = fmaxf(mx, pv[u]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mold = rm[row], mnew = fmaxf(mold, mx);
            float l = 0.0f;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const float p = pv[u] == -INFINITY ? 0.0f : exp2f(pv[u] - mnew);
                ps[row][lane + 32 * u] = __float2half_rn(p);
                l += p;
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float alpha = exp2f(mold - mnew);
                ra[row] = alpha;
                rl[row] = fmaf(rl[row], alpha, l);
                rm[row] = mnew;
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w the group w / WPG's dims (w % WPG) HD / WPG ..
        {
            const int gr = warp / WPG, sl = warp - gr * WPG;
            const float ag = ra[16 * gr + g], ag8 = ra[16 * gr + g + 8];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                acc[n][0] *= ag;
                acc[n][1] *= ag;
                acc[n][2] *= ag8;
                acc[n][3] *= ag8;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &ps[16 * gr + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, &vs[kk + (lane & 15)][sl * (HD / WPG) + n * 8]);
                    mma16816(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();   // the shared tiles are free for the next chunk
    }
    {
        const int gr = warp / WPG, sl = warp - gr * WPG;
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int m = 16 * gr + g + 8 * hr;
            if (m >= M) continue;
            float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
            if (sl == 0 && c == 0) {
                out[0] = rm[m];
                out[1] = rl[m];
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int d = sl * (HD / WPG) + n * 8 + 2 * c;
                out[2 + d] = acc[n][2 * hr];
                out[3 + d] = acc[n][2 * hr + 1];
            }
        }
    }
}

// the K8V4 kernel's launch: one block a (split, KV head, segment), W warps, its shared bytes opted in once per
// instantiation and device
template <int HD, int R, int W>
void launch_k8v4(const float* q, int n_head, int n_kv, const DecSegs& segs, float scale, int gx, int n_chunks, float* scratch,
                 cudaStream_t s) {
    const size_t smem = dec_k8_smem<HD, R>();
    static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
    const int dev = opt_in_device();
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(decode_k8v4_kernel<HD, R, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
            cudaSuccess) {
            halo::log::error("halo", "gqa decode attention: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    launch_pdl(decode_k8v4_kernel<HD, R, W>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.n), dim3(W * 32), smem, s, q,
               n_head, n_kv, segs, n_chunks, scale, scratch);
}

// What a decode launch runs: the engine's choice (the K8V4 kernel while the window fits 48 rows, the tiled launch
// otherwise), or, for the parity tests and the bench only, one of the kernels the engine does not select.
enum class DecPath { kEngine, kTiled, kWindow };

// A launch takes one block a (split, KV head) with every row of the window: the K8V4 kernel for a K8V4 cache with
// M <= 48 rows, the 16-row-tile launch past that (and for fp16 and int8 caches).  A rotated q4_0 cache (--kv q4_0)
// takes the Q4 instantiations of the tiled and window kernels; every segment of a launch holds one format.
template <int HD>
void launch_segs(const float* q, int n_head, int n_kv, DecSegs& segs, float* scratch, int n_chunks, float* out, cudaStream_t s,
                 DecPath path, const float* gate_q = nullptr, uint16_t* gated16 = nullptr) {
    const float scale = 1.4426950408889634f / sqrtf((float) HD);
    const int gx = decode_splits(n_chunks);
    const int G = n_head / n_kv;
    int mmax = 0;
    for (int k = 0; k < segs.n; ++k) mmax = std::max(mmax, (segs.row0[k + 1] - segs.row0[k]) * G);
    bool k8v4 = path == DecPath::kEngine;   // every segment's cache, or the tiled launch
    if constexpr (HD != 256) k8v4 = false;
    for (int k = 0; k < segs.n && k8v4; ++k) k8v4 = segs.k8v4[k].k != nullptr;
    if (k8v4 && mmax <= 48) {
        if constexpr (HD == 256) {
            if (mmax <= 16) launch_k8v4<HD, 1, 8>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
            else if (mmax <= 32) launch_k8v4<HD, 2, 8>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
            else launch_k8v4<HD, 3, 12>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
        }
    } else if (segs.q4[0].k != nullptr) {
        if (path == DecPath::kWindow && mmax <= 48) {
            if (mmax <= 16) launch_win<HD, 1, true>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
            else if (mmax <= 32) launch_win<HD, 2, true>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
            else launch_win<HD, 3, true>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
        } else {
            for (int k = 0; k < segs.n; ++k)
                segs.tile0[k + 1] = segs.tile0[k] + ((segs.row0[k + 1] - segs.row0[k]) * G + 15) / 16;
            launch_pdl(decode_mma_kernel<HD, true>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.tile0[segs.n]), dim3(128),
                       0, s, q, n_head, n_kv, segs, n_chunks, scale, scratch);
        }
    } else if (path == DecPath::kWindow && mmax <= 48) {
        if (mmax <= 16) launch_win<HD, 1>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
        else if (mmax <= 32) launch_win<HD, 2>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
        else launch_win<HD, 3>(q, n_head, n_kv, segs, scale, gx, n_chunks, scratch, s);
    } else {
        for (int k = 0; k < segs.n; ++k)
            segs.tile0[k + 1] = segs.tile0[k] + ((segs.row0[k + 1] - segs.row0[k]) * G + 15) / 16;
        launch_pdl(decode_mma_kernel<HD>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) segs.tile0[segs.n]), dim3(128), 0, s,
                   q, n_head, n_kv, segs, n_chunks, scale, scratch);
    }
    check("gqa decode attention");
    launch_pdl(decode_merge_kernel<HD>, dim3((unsigned) n_head, (unsigned) segs.row0[segs.n]), dim3(HD), 0, s,
               (const float*) scratch, n_head, gx, out, gate_q, gated16, segs);
    check("gqa decode attention/merge");
}

template <int HD>
void launch_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_kv, const Range& r, int T,
                   float* scratch, int n_chunks, float* out, cudaStream_t s, DecPath path, KvInt8 q8 = {},
                   const float* gate_q = nullptr, uint16_t* gated16 = nullptr, KvFar far_kv = {}, KvK8V4 k8v4 = {},
                   KvQ4 q4 = {}) {
    DecSegs segs;
    segs.row0[1] = T;
    segs.kc[0] = (const __half*) kc;
    segs.vc[0] = (const __half*) vc;
    segs.q8[0] = q8;
    segs.far_kv[0] = far_kv;
    segs.k8v4[0] = k8v4;
    segs.q4[0] = q4;
    segs.r[0] = r;
    launch_segs<HD>(q, n_head, n_kv, segs, scratch, n_chunks, out, s, path, gate_q, gated16);
}

__device__ __forceinline__ void mma16832_u8s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ float ex2(float x) {   // 2^x, -inf -> 0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---------------------------------------------------------------- prompt, fp16 cache
// Block (query tile, KV head): its 96 rows are the tile's positions times the KV group's query heads (head fastest), so
// a K/V tile is read once for every head of the group.  Warp w owns rows 16 w .. 16 w + 15.  Cells stream in tiles of
// 32.  QK^T on fp16 tensor cores with fp32 accumulation; P V on fp16 tensor cores accumulating each 32-cell tile in
// fp16, added to the fp32 output.  Online softmax in fp32.
constexpr int PNW = 6, PROWS = 16 * PNW, PBN = 32;
constexpr int LDH = HD + 8;    // padded fp16 row (halves)

struct PromptSmem16 {
    __half q[PROWS][LDH];
    __half k[PBN][LDH];
    __half v[PBN][LDH];
};

__global__ void __launch_bounds__(PNW * 32) prompt_f16_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                              const __half* __restrict__ vc, int n_head, int n_kv,
                                                              int64_t pos0, int64_t T, float* __restrict__ out, KvFar far_kv) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PromptSmem16& sm = *reinterpret_cast<PromptSmem16*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = n_head / n_kv, PB = PROWS / G;   // query heads per KV head, positions per tile
    const int kvh = blockIdx.y;
    const int64_t q0 = (int64_t) (gridDim.x - 1 - blockIdx.x) * PB;   // the longest tiles first
    const int64_t last_pos = min(T, q0 + PB) - 1;
    const int64_t cells = pos0 + last_pos + 1;
    const int g = lane >> 2, tq = lane & 3;
    const int ra = warp * 16 + g, rb = ra + 8;                         // this thread's two rows
    const int64_t pos_a = q0 + ra / G, pos_b = q0 + rb / G;
    const int head_a = kvh * G + ra % G, head_b = kvh * G + rb % G;

    for (int i = tid; i < PROWS * HD / 4; i += PNW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        const int64_t pos = q0 + r / G;
        const int head = kvh * G + r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pos < T) v = *reinterpret_cast<const float4*>(q + (pos * n_head + head) * HD + c);
        *reinterpret_cast<__half2*>(&sm.q[r][c]) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(&sm.q[r][c + 2]) = __floats2half2_rn(v.z, v.w);
    }

    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int n_tiles = (int) ((cells + PBN - 1) / PBN);

    for (int ti = 0; ti < n_tiles; ++ti) {
        const int64_t c0 = (int64_t) ti * PBN;
        __syncthreads();   // the previous tile is consumed (and the query tile is written)
        for (int i = tid; i < PBN * HD / 8; i += PNW * 32) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = c0 + r < cells;
            int64_t cell = ok ? c0 + r : 0;
            const bool fa = kv_far(far_kv, cell);
            const int64_t row = cell * n_kv + kvh;
            cp16(&sm.k[r][c], (fa ? (const __half*) far_kv.k : kc) + row * HD + c, ok);
            cp16(&sm.v[r][c], (fa ? (const __half*) far_kv.v : vc) + row * HD + c, ok);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // S = Q K^T for the warp's 16 rows and the tile's 32 cells (four 8-cell tiles)
        float s[PBN / 8][4];
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < HD; k += 16) {
            unsigned a0, a1, a2, a3;
            ldsm_x4(a0, a1, a2, a3, &sm.q[warp * 16 + (lane & 15)][k + (lane >> 4) * 8]);
#pragma unroll
            for (int n = 0; n < PBN / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sm.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], a0, a1, a2, a3, b0, b1);
                mma16816(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        // mask and online softmax: thread holds rows ra (s[n][0..1]) and rb (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t cell = c0 + n * 8 + tq * 2 + (i & 1);
                const int64_t pos = i < 2 ? pos_a : pos_b;
                const bool vis = cell <= pos0 + pos && cell < cells;
                s[n][i] = vis ? s[n][i] * kScaleLog2 : kNeg;
                if (i < 2) mx_a = fmaxf(mx_a, s[n][i]);
                else mx_b = fmaxf(mx_b, s[n][i]);
            }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float sc_a = exp2f(m_a - mx_a), sc_b = exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) {
            s[n][0] = s[n][0] <= kNeg ? 0.0f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] <= kNeg ? 0.0f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] <= kNeg ? 0.0f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] <= kNeg ? 0.0f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * sc_a + ps_a;
        l_b = l_b * sc_b + ps_b;
        // O = O * scale + P V, the tile's P V summed in fp16 on the tensor cores
        unsigned pa[PBN / 16][4];
#pragma unroll
        for (int j = 0; j < PBN / 16; ++j) {
            pa[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pa[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pa[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pa[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < HD / 8; n += 2) {
            unsigned d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < PBN / 16; ++j) {
                unsigned b0, b1, b2, b3;
                // V rows are cells (k), columns dims (n): transposed loads give the col-major B of dims n*8.., n*8+8..
                ldsm_x4_t(b0, b1, b2, b3, &sm.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816_h(d0, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b0, b1);
                mma16816_h(d1, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b2, b3);
            }
            const float2 e0 = __half22float2(*reinterpret_cast<const __half2*>(&d0[0]));
            const float2 e1 = __half22float2(*reinterpret_cast<const __half2*>(&d0[1]));
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d1[0]));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1[1]));
            o[n][0] = fmaf(o[n][0], sc_a, e0.x); o[n][1] = fmaf(o[n][1], sc_a, e0.y);
            o[n][2] = fmaf(o[n][2], sc_b, e1.x); o[n][3] = fmaf(o[n][3], sc_b, e1.y);
            o[n + 1][0] = fmaf(o[n + 1][0], sc_a, f0.x); o[n + 1][1] = fmaf(o[n + 1][1], sc_a, f0.y);
            o[n + 1][2] = fmaf(o[n + 1][2], sc_b, f1.x); o[n + 1][3] = fmaf(o[n + 1][3], sc_b, f1.y);
        }
    }
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int col = n * 8 + tq * 2;
        if (pos_a < T) *reinterpret_cast<float2*>(out + (pos_a * n_head + head_a) * HD + col) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (pos_b < T) *reinterpret_cast<float2*>(out + (pos_b * n_head + head_b) * HD + col) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

// ---------------------------------------------------------------- prompt, int8 cache
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.  The structure follows
// Infernix's fast int8 prompt kernel (src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh); Halocline's
// cache is unrotated and unpaged, so the queries are rounded to int8 per 64 dims directly and a key tile is gathered
// cell by cell.
//
// Block (query tile, query head), longest tiles first: warp w owns the tile's positions 16 w .. 16 w + 15 of one head
// for the whole sweep, so scores, probabilities and the 256-wide output stay in registers and the only block barrier
// publishes a key tile.  Keys stream in tiles of 64 cells, K and V double-buffered as raw int8 codes with their scales
// (16-byte pieces XOR-swizzled by the row).  QK^T: int8 tensor cores, each 64-dim group's integer sum times the query's
// and the cell's scales in fp32.  P V in 8 bits (Infernix's prompt-pv8): per 64-dim V group, each probability times
// its cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum); u8 x s8
// tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the fp32 output.
// Online softmax in fp32.
//
// A split launch (gridDim.z > 1) gives each block a contiguous run of its key tiles; the block writes its rows
// normalized by its own sum with their (max, sum) and prompt_i8_merge_kernel combines the splits.
constexpr int I8BC = 64;                                   // cells per key tile
constexpr int I8_TILE = I8BC * HD;                         // bytes of one tile's K (or V) codes
constexpr int I8_STAGE = 2 * I8_TILE + 2 * I8BC * 4 * 2;   // K, V codes and their fp16 scales

template <int W> constexpr int i8_smem() { return W * 16 * HD + 2 * I8_STAGE; }

// IO: the queries and the output rows in FP32 or BF16 (the prompt chunk's)
__device__ __forceinline__ float io_ld(const float* p) { return *p; }
__device__ __forceinline__ float io_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void io_st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void io_st4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
}
__device__ __forceinline__ void io_st1(float* p, float v) { *p = v; }
__device__ __forceinline__ void io_st1(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <int W, bool Split, class IO = float>
__global__ void __launch_bounds__(W * 32, 1) prompt_i8_kernel(const IO* __restrict__ q, int n_head, int n_kv, int pos0,
                                                              int T, IO* __restrict__ out, KvInt8 q8,
                                                              float* __restrict__ part_rows, float2* __restrict__ part_stats,
                                                              KvFar far_kv) {
    constexpr int Threads = W * 32, Br = W * 16;
    constexpr int G4 = HD / 64;           // scale groups per row
    constexpr int NT = I8BC / 8;          // score n8 tiles per key tile
    constexpr int KS = I8BC / 16;         // P V k-steps per key tile
    constexpr int DB = HD / 16;           // 16-dim output blocks
    static_assert(I8_TILE / 16 % Threads == 0 && 2 * I8BC <= Threads, "int8 prompt attention: tile copy split");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* q_s = reinterpret_cast<int8_t*>(smem_raw);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;   // the first n_head blocks take every head's last tile
    const int head = linear % n_head;
    const int q0 = ((int) gridDim.x - 1 - linear / n_head) * Br;
    const int kvh = head / (n_head / n_kv);
    const int rows = min(Br, T - q0);
    const int max_qabs = pos0 + q0 + rows - 1;
    const int key_tiles = max_qabs / I8BC + 1;
    const int split = Split ? (int) blockIdx.z : 0;
    const int per_split = Split ? (key_tiles + (int) gridDim.z - 1) / (int) gridDim.z : 0;
    const int kb_begin = Split ? min(key_tiles, split * per_split) : 0;
    const int kb_end = Split ? min(key_tiles, kb_begin + per_split) : key_tiles;

    // each warp rounds its own 16 rows to int8 per 64 dims (amax / 127); a thread keeps the scales of its two C rows
    float qsr[2][G4];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = warp * 16 + r;
        const bool valid = row < rows;
        const IO* src = q + ((int64_t) (q0 + row) * n_head + head) * HD;
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = valid ? io_ld(src + lane + 32 * k) : 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            const float amax = warp_max(fmaxf(fabsf(x[2 * grp]), fabsf(x[2 * grp + 1])));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = grp * 64 + h * 32 + lane;
                const int c = max(-127, min(127, __float2int_rn(x[2 * grp + h] * inv)));
                q_s[row * HD + (((d >> 4) ^ (row & 7)) << 4) + (d & 15)] = (int8_t) c;
            }
            if (g == (r & 7)) qsr[r >> 3][grp] = amax / 127.0f;
        }
    }

    // a key tile: 256-byte K and V rows per cell (the cache interleaves the KV heads), cells past the block's last
    // visible one zero-filled so masked columns stay finite
    const auto issue_tile = [&](int kb) {
        unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        int8_t* k_s = reinterpret_cast<int8_t*>(base);
        int8_t* v_s = k_s + I8_TILE;
        uint16_t* ks_s = reinterpret_cast<uint16_t*>(v_s + I8_TILE);
        uint16_t* vs_s = ks_s + I8BC * G4;
        const int valid = min(I8BC, max_qabs + 1 - kb * I8BC);
#pragma unroll
        for (int i = 0; i < I8_TILE / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key = chunk >> 4, c = chunk & 15;
            const bool ok = key < valid;
            int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const KvInt8& c8 = kv_far(far_kv, cell) ? far_kv.q8 : q8;
            const int64_t src = (cell * n_kv + kvh) * HD + c * 16;
            const int dst = key * HD + ((c ^ (key & 7)) << 4);
            cp16(k_s + dst, c8.k + src, ok);
            cp16(v_s + dst, c8.v + src, ok);
        }
        if (tid < 2 * I8BC) {
            const int key = tid & (I8BC - 1);
            const bool ok = key < valid;
            int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const KvInt8& c8 = kv_far(far_kv, cell) ? far_kv.q8 : q8;
            const int64_t src = (cell * n_kv + kvh) * G4;
            if (tid < I8BC) cp8(ks_s + key * G4, c8.ks + src, ok);
            else cp8(vs_s + key * G4, c8.vs + src, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = warp * 16 + a_rin + ((a_mat & 1) << 3);
    const int a_chunk = a_mat >> 1;        // 16-byte piece within a 32-byte k step
    const int b_chunk = (lane >> 3) & 1;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool warp_active = warp * 16 < rows;
    const int warp_min_qabs = pos0 + q0 + warp * 16;
    const int warp_max_qabs = pos0 + q0 + min(warp * 16 + 15, rows - 1);
    const int qabs0 = row0 < rows ? pos0 + q0 + row0 : -1;
    const int qabs1 = row1 < rows ? pos0 + q0 + row1 : -1;

    float acc[DB][2][4];
#pragma unroll
    for (int b = 0; b < DB; ++b)
#pragma unroll
        for (int p = 0; p < 2; ++p) acc[b][p][0] = acc[b][p][1] = acc[b][p][2] = acc[b][p][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (kb_begin < kb_end) issue_tile(kb_begin);
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();   // tile kb is in, and every warp is done with the other stage
        if (kb + 1 < kb_end) issue_tile(kb + 1);
        const int k0 = kb * I8BC;
        if (!warp_active || k0 > warp_max_qabs) continue;
        const unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        const int8_t* k_s = reinterpret_cast<const int8_t*>(base);
        const int8_t* v_s = k_s + I8_TILE;
        const uint16_t* ks_s = reinterpret_cast<const uint16_t*>(v_s + I8_TILE);
        const uint16_t* vs_s = ks_s + I8BC * G4;

        // S = Q K^T: 16 rows x 64 cells
        float s[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            unsigned af[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int ch = (grp * 2 + kk) * 2 + a_chunk;
                ldsm_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3], q_s + a_row * HD + ((ch ^ (a_row & 7)) << 4));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int c[4] = {0, 0, 0, 0};
                const int key = n * 8 + a_rin;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int ch = (grp * 2 + kk) * 2 + b_chunk;
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, k_s + key * HD + ((ch ^ (key & 7)) << 4));
                    mma16832_s8(c, af[kk][0], af[kk][1], af[kk][2], af[kk][3], b0, b1);
                }
                const int ka = n * 8 + 2 * tq;
                const float ks0 = h2f(ks_s[ka * G4 + grp]), ks1 = h2f(ks_s[(ka + 1) * G4 + grp]);
                s[n][0] = fmaf(qsr[0][grp] * ks0, (float) c[0], s[n][0]);
                s[n][1] = fmaf(qsr[0][grp] * ks1, (float) c[1], s[n][1]);
                s[n][2] = fmaf(qsr[1][grp] * ks0, (float) c[2], s[n][2]);
                s[n][3] = fmaf(qsr[1][grp] * ks1, (float) c[3], s[n][3]);
            }
        }

        // mask (only tiles crossing the warp's diagonal) and online softmax, scores kept unscaled
        const bool full = k0 + I8BC - 1 <= warp_min_qabs;
        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (!full) {
                const int c0 = k0 + n * 8 + 2 * tq, c1 = c0 + 1;
                s[n][0] = c0 <= qabs0 ? s[n][0] : -INFINITY;
                s[n][1] = c1 <= qabs0 ? s[n][1] : -INFINITY;
                s[n][2] = c0 <= qabs1 ? s[n][2] : -INFINITY;
                s[n][3] = c1 <= qabs1 ? s[n][3] : -INFINITY;
            }
            bm0 = fmaxf(bm0, fmaxf(s[n][0], s[n][1]));
            bm1 = fmaxf(bm1, fmaxf(s[n][2], s[n][3]));
        }
        const float nm0 = fmaxf(m0, quad_max(bm0)), nm1 = fmaxf(m1, quad_max(bm1));
        const float ns0 = nm0 == -INFINITY ? 0.0f : nm0 * kScaleLog2;
        const float ns1 = nm1 == -INFINITY ? 0.0f : nm1 * kScaleLog2;
        const float al0 = m0 == -INFINITY ? 0.0f : ex2(m0 * kScaleLog2 - ns0);
        const float al1 = m1 == -INFINITY ? 0.0f : ex2(m1 * kScaleLog2 - ns1);
        m0 = nm0;
        m1 = nm1;
        // the probabilities become the P V A fragments: k step j covers cells 16 j .. 16 j + 15 (score tiles 2 j, 2 j + 1)
        unsigned pa[KS][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float p00 = ex2(fmaf(s[n][0], kScaleLog2, -ns0)), p01 = ex2(fmaf(s[n][1], kScaleLog2, -ns0));
            const float p10 = ex2(fmaf(s[n][2], kScaleLog2, -ns1)), p11 = ex2(fmaf(s[n][3], kScaleLog2, -ns1));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[n >> 1][(n & 1) * 2] = pack_h2(p00, p01);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_h2(p10, p11);
        }
        l0 = fmaf(l0, al0, bl0);   // lane-partial; the quad is summed after the sweep
        l1 = fmaf(l1, al1, bl1);

        // 8-bit P V (Infernix's prompt-pv8, fast_tiled_mma.cuh): per 64-dim V group, each probability times its
        // cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum);
        // u8 x s8 tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the
        // fp32 output.  A k32 step takes the lane's own cells {2t, 2t+1, 8+2t, 9+2t} (+16), as pa holds them.
        const __half2 origin = __float2half2_rn(1024.0f);
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            __half2 prod[KS][4];
            __half2 mx0 = __float2half2_rn(0.0f), mx1 = mx0;
#pragma unroll
            for (int j = 0; j < KS; ++j) {
                const int key = j * 16 + 2 * tq;
                const __half2 lo = __halves2half2(__ushort_as_half(vs_s[key * G4 + grp]), __ushort_as_half(vs_s[(key + 1) * G4 + grp]));
                const __half2 hi = __halves2half2(__ushort_as_half(vs_s[(key + 8) * G4 + grp]),
                                                  __ushort_as_half(vs_s[(key + 9) * G4 + grp]));
                prod[j][0] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][0]), lo);
                prod[j][1] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][1]), lo);
                prod[j][2] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][2]), hi);
                prod[j][3] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][3]), hi);
                mx0 = __hmax2(mx0, __hmax2(prod[j][0], prod[j][2]));
                mx1 = __hmax2(mx1, __hmax2(prod[j][1], prod[j][3]));
            }
            const float rm0 = quad_max(fmaxf(__low2float(mx0), __high2float(mx0)));
            const float rm1 = quad_max(fmaxf(__low2float(mx1), __high2float(mx1)));
            // codes rint(product * inverse), the fp16 inverse rounded down so none passes 255; the step is its reciprocal
            const __half inv0 = __float2half_rd(rm0 > 0.0f ? 255.0f / rm0 : 0.0f);
            const __half inv1 = __float2half_rd(rm1 > 0.0f ? 255.0f / rm1 : 0.0f);
            const float step0 = __hgt(inv0, __float2half(0.0f)) ? 1.0f / __half2float(inv0) : 0.0f;
            const float step1 = __hgt(inv1, __float2half(0.0f)) ? 1.0f / __half2float(inv1) : 0.0f;
            const __half2 i02 = __half2half2(inv0), i12 = __half2half2(inv1);
            unsigned a8[2][4];   // fp16 1024 + n holds n in its low byte
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                unsigned qv[2][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * st + h;
                    const __half2 c0 = __hfma2(prod[j][0], i02, origin), c1 = __hfma2(prod[j][1], i12, origin);
                    const __half2 c2 = __hfma2(prod[j][2], i02, origin), c3 = __hfma2(prod[j][3], i12, origin);
                    qv[h][0] = *reinterpret_cast<const unsigned*>(&c0);
                    qv[h][1] = *reinterpret_cast<const unsigned*>(&c1);
                    qv[h][2] = *reinterpret_cast<const unsigned*>(&c2);
                    qv[h][3] = *reinterpret_cast<const unsigned*>(&c3);
                }
                a8[st][0] = __byte_perm(qv[0][0], qv[0][2], 0x6420);
                a8[st][1] = __byte_perm(qv[0][1], qv[0][3], 0x6420);
                a8[st][2] = __byte_perm(qv[1][0], qv[1][2], 0x6420);
                a8[st][3] = __byte_perm(qv[1][1], qv[1][3], 0x6420);
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int db = grp * 4 + b;
                int ev[4] = {0, 0, 0, 0}, od[4] = {0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    // matrices: cells 32 st + 0..7, 8..15, 16..23, 24..31 of d block db; lane (g, t) gets cells
                    // {2t, 2t+1} at dims {2g, 2g+1}
                    const int key = st * 32 + (a_mat << 3) + a_rin;
                    unsigned r[4];
                    ldsm_x4_t(r[0], r[1], r[2], r[3], v_s + key * HD + ((db ^ (key & 7)) << 4));
                    mma16832_u8s8(ev, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x6420),
                                  __byte_perm(r[2], r[3], 0x6420));
                    mma16832_u8s8(od, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x7531),
                                  __byte_perm(r[2], r[3], 0x7531));
                }
                float (&e)[4] = acc[db][0];
                float (&o)[4] = acc[db][1];
                e[0] = fmaf(e[0], al0, step0 * (float) ev[0]);
                e[1] = fmaf(e[1], al0, step0 * (float) ev[1]);
                e[2] = fmaf(e[2], al1, step1 * (float) ev[2]);
                e[3] = fmaf(e[3], al1, step1 * (float) ev[3]);
                o[0] = fmaf(o[0], al0, step0 * (float) od[0]);
                o[1] = fmaf(o[1], al0, step0 * (float) od[1]);
                o[2] = fmaf(o[2], al1, step1 * (float) od[2]);
                o[3] = fmaf(o[3], al1, step1 * (float) od[3]);
            }
        }
    }

    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    // even n8 tiles hold dims 16 b + 2 n, odd ones 16 b + 2 n + 1: lane (g, tq) owns dims 16 b + 4 tq .. + 3 of its rows
    const auto store = [&](int row, int e, float m, float l, float inv) {
        if (row >= rows) return;
        const int64_t idx = Split ? ((int64_t) split * T + q0 + row) * n_head + head : (int64_t) (q0 + row) * n_head + head;
        if (Split && tq == 0) part_stats[idx] = make_float2(m, l);
#pragma unroll
        for (int b = 0; b < DB; ++b) {
            const float4 v = make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv);
            if (Split) io_st4(part_rows + idx * HD + b * 16 + 4 * tq, v);   // the splits' rows stay FP32
            else io_st4(out + idx * HD + b * 16 + 4 * tq, v);
        }
    };
    store(row0, 0, m0, l0, i0);
    store(row1, 2, m1, l1, i1);
}

// ---------------------------------------------------------------- prompt, block caches (K8V4 and Q4)
// The int8 prompt kernel's shape over a cache whose V is rotated q4_0 (halo/kernels/kv_q4_layout.hpp), K either the int8
// cache's codes with a scale a 64 values (K8V4, `K4` false) or rotated q4_0 with a scale a 32-dim block (Q4, `K4` true).
// V's blocks are 32 dims wide and each carries its own fp16 scale, so a key tile's V arrives as int8 codes (code - 8,
// exact) with one scale per cell and 32-dim block - the K/V tile bytes stay 16-bit-per-value pieces, which is what lets
// the u8 x s8 P V below reuse the int8 kernel's fragment layout: each 32-dim block's probabilities are re-quantized to
// u8 against that row's largest product over the tile, the mma sums the stored codes exactly in int32, and the partial
// times the row's step joins the fp32 output. Q4's K tile is expanded the same way, and its QK step keeps one int32
// accumulator a 32-dim block (the `kk` loop's two steps) so each block is scaled by its own delta. A 4-warp block: with
// eight V scales a cell the 8-warp shape's shared memory no longer fits the card's 99 KiB.
constexpr int V4NG = HD / kKvQ4Block;   // q4_0 blocks per 256-wide row: V's eight, and Q4's K
constexpr int K8V4_STAGE = 2 * I8_TILE + I8BC * (HD / 64) * 2 + I8BC * V4NG * 2;
constexpr int Q4_STAGE = 2 * I8_TILE + I8BC * V4NG * 2 + I8BC * V4NG * 2;

template <int W, bool K4> constexpr int v4_smem() { return W * 16 * HD + 2 * (K4 ? Q4_STAGE : K8V4_STAGE); }
static_assert(v4_smem<8, false>() <= 101376, "K8V4 prompt attention: the 8-warp shape must fit sm_89's 99 KiB opt-in");
static_assert(v4_smem<4, true>() <= 101376 && v4_smem<8, true>() > 101376, "Q4 prompt attention: 4 warps only");

// 16 dims of a q4_0 row as the int8 kernel's swizzled tile piece: dims 16 c .. 16 c + 15 are in block c / 2 at
// (c & 1) * 16, stored as the code less 8 (exact, -8 .. 7), and zero past the tile's last valid cell
//
// `abs_scale`: a q4_0 scale is signed (d = the largest-magnitude value over -8), but V's P V folds the scale into the
// probabilities and re-quantizes them to u8, which needs a scale >= 0. With it set, a block with d < 0 has its codes
// negated here, (code - 8) * d = (8 - code) * |d|, exact and within s8 (-7 .. 8), and the tile keeps |d| (issue_tile).
// K's scales multiply an int32 dot product in fp32, so K keeps its signed scale and plain codes.
__device__ __forceinline__ void q4_piece(const uint8_t* blocks, int c, int8_t* dst, bool ok, bool abs_scale) {
    const KvQ4Block* blk = reinterpret_cast<const KvQ4Block*>(blocks) + (c >> 1);
    int codes[16];
    kv_q4_codes8(blk->qs, (c & 1) * 16, codes);
    kv_q4_codes8(blk->qs, (c & 1) * 16 + 8, codes + 8);
    const bool neg = abs_scale && (blk->d & 0x8000u) != 0;
    union { int32_t word[4]; int8_t byte[16]; } pack;
#pragma unroll
    for (int e = 0; e < 16; ++e) pack.byte[e] = (int8_t) (neg ? 8 - codes[e] : codes[e] - 8);
#pragma unroll
    for (int e = 0; e < 4; ++e) *reinterpret_cast<int*>(dst + 4 * e) = ok ? pack.word[e] : 0;
}

template <int W, bool Split, bool K4, class IO = float>
__global__ void __launch_bounds__(W * 32, 1) prompt_v4_kernel(const IO* __restrict__ q, int n_head, int n_kv, int pos0,
                                                             int T, IO* __restrict__ out, KvK8V4 k8v4, KvQ4 q4,
                                                             float* __restrict__ part_rows, float2* __restrict__ part_stats) {
    constexpr int Threads = W * 32, Br = W * 16;
    constexpr int G4 = HD / 64;           // K's 64-value groups in the int8 form
    constexpr int NG = V4NG;              // V's q4_0 blocks, and Q4's K
    constexpr int KSG = K4 ? NG : G4;     // K's scales a row
    constexpr int STAGE = K4 ? Q4_STAGE : K8V4_STAGE;
    constexpr int NT = I8BC / 8;          // score n8 tiles per key tile
    constexpr int KS = I8BC / 16;         // P V k-steps per key tile
    constexpr int DB = HD / 16;           // 16-dim output blocks
    static_assert(I8_TILE / 16 % Threads == 0, "block-cache prompt attention: tile copy split");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* q_s = reinterpret_cast<int8_t*>(smem_raw);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;   // the first n_head blocks take every head's last tile
    const int head = linear % n_head;
    const int q0 = ((int) gridDim.x - 1 - linear / n_head) * Br;
    const int kvh = head / (n_head / n_kv);
    const int rows = min(Br, T - q0);
    const int max_qabs = pos0 + q0 + rows - 1;
    const int key_tiles = max_qabs / I8BC + 1;
    const int split = Split ? (int) blockIdx.z : 0;
    const int per_split = Split ? (key_tiles + (int) gridDim.z - 1) / (int) gridDim.z : 0;
    const int kb_begin = Split ? min(key_tiles, split * per_split) : 0;
    const int kb_end = Split ? min(key_tiles, kb_begin + per_split) : key_tiles;

    // the queries: the int8 kernel's per-64-dim rounding, the same codes and scales
    float qsr[2][G4];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = warp * 16 + r;
        const bool valid = row < rows;
        const IO* src = q + ((int64_t) (q0 + row) * n_head + head) * HD;
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = valid ? io_ld(src + lane + 32 * k) : 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            const float amax = warp_max(fmaxf(fabsf(x[2 * grp]), fabsf(x[2 * grp + 1])));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = grp * 64 + h * 32 + lane;
                const int c = max(-127, min(127, __float2int_rn(x[2 * grp + h] * inv)));
                q_s[row * HD + (((d >> 4) ^ (row & 7)) << 4) + (d & 15)] = (int8_t) c;
            }
            if (g == (r & 7)) qsr[r >> 3][grp] = amax / 127.0f;
        }
    }

    // a key tile: K's codes (the int8 form's, or expanded from its q4_0 blocks) and V's, the q4_0 blocks expanded to
    // (code - 8) in the int8 kernel's swizzled pieces, and one fp16 scale a cell for each of K's groups or blocks and
    // each of V's 32-value blocks
    const auto issue_tile = [&](int kb) {
        unsigned char* base = smem_raw + Br * HD + (kb & 1) * STAGE;
        int8_t* k_s = reinterpret_cast<int8_t*>(base);
        int8_t* v_s = k_s + I8_TILE;
        uint16_t* ks_s = reinterpret_cast<uint16_t*>(v_s + I8_TILE);
        uint16_t* vs_s = ks_s + I8BC * KSG;
        const int valid = min(I8BC, max_qabs + 1 - kb * I8BC);
#pragma unroll
        for (int i = 0; i < I8_TILE / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key = chunk >> 4, c = chunk & 15;
            const bool ok = key < valid;
            int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const int64_t row = cell * n_kv + kvh;
            const int dst = key * HD + ((c ^ (key & 7)) << 4);
            if constexpr (K4) q4_piece(q4.k + row * (int64_t) kv_q4_head_bytes(HD), c, k_s + dst, ok, false);
            else cp16(k_s + dst, k8v4.k + row * HD + c * 16, ok);
            // V's codes carry their block's sign (q4_piece's `abs_scale`): the P V folds the scale into u8 probabilities
            if constexpr (K4) q4_piece(q4.v + row * (int64_t) kv_q4_head_bytes(HD), c, v_s + dst, ok, true);
            else q4_piece(k8v4.vq + row * (int64_t) kv_q4_head_bytes(HD), c, v_s + dst, ok, true);
        }
        for (int i = tid; i < I8BC * KSG; i += Threads) {
            const int key = i / KSG, g = i % KSG;
            const bool ok = key < valid;
            const int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const int64_t row = cell * n_kv + kvh;
            if constexpr (K4) {
                const KvQ4Block* blk = reinterpret_cast<const KvQ4Block*>(q4.k + row * (int64_t) kv_q4_head_bytes(HD)) + g;
                ks_s[i] = ok ? blk->d : (uint16_t) 0;
            } else {
                ks_s[i] = ok ? k8v4.ks[row * G4 + g] : (uint16_t) 0;
            }
        }
        for (int i = tid; i < I8BC * NG; i += Threads) {
            const int key = i / NG, b = i % NG;
            const bool ok = key < valid;
            const int64_t cell = (int64_t) kb * I8BC + (ok ? key : 0);
            const int64_t row = cell * n_kv + kvh;
            const KvQ4Block* blk =
                reinterpret_cast<const KvQ4Block*>(K4 ? q4.v + row * (int64_t) kv_q4_head_bytes(HD)
                                                     : k8v4.vq + row * (int64_t) kv_q4_head_bytes(HD)) + b;
            vs_s[i] = ok ? (uint16_t) (blk->d & 0x7FFFu) : (uint16_t) 0;   // |d|: the codes carry the sign (issue_tile)
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = warp * 16 + a_rin + ((a_mat & 1) << 3);
    const int a_chunk = a_mat >> 1;        // 16-byte piece within a 32-byte k step
    const int b_chunk = (lane >> 3) & 1;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool warp_active = warp * 16 < rows;
    const int warp_min_qabs = pos0 + q0 + warp * 16;
    const int warp_max_qabs = pos0 + q0 + min(warp * 16 + 15, rows - 1);
    const int qabs0 = row0 < rows ? pos0 + q0 + row0 : -1;
    const int qabs1 = row1 < rows ? pos0 + q0 + row1 : -1;

    float acc[DB][2][4];
#pragma unroll
    for (int b = 0; b < DB; ++b)
#pragma unroll
        for (int p = 0; p < 2; ++p) acc[b][p][0] = acc[b][p][1] = acc[b][p][2] = acc[b][p][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (kb_begin < kb_end) issue_tile(kb_begin);
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();   // tile kb is in, and every warp is done with the other stage
        if (kb + 1 < kb_end) issue_tile(kb + 1);
        const int k0 = kb * I8BC;
        if (!warp_active || k0 > warp_max_qabs) continue;
        const unsigned char* base = smem_raw + Br * HD + (kb & 1) * STAGE;
        const int8_t* k_s = reinterpret_cast<const int8_t*>(base);
        const int8_t* v_s = k_s + I8_TILE;
        const uint16_t* ks_s = reinterpret_cast<const uint16_t*>(v_s + I8_TILE);
        const uint16_t* vs_s = ks_s + I8BC * KSG;

        // S = Q K^T: 16 rows x 64 cells, the int8 kernel's int8 tensor cores; K4 keeps a block's 32-dim half in its own
        // accumulator so each half is scaled by its own q4_0 delta
        float s[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            unsigned af[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int ch = (grp * 2 + kk) * 2 + a_chunk;
                ldsm_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3], q_s + a_row * HD + ((ch ^ (a_row & 7)) << 4));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int ca[4] = {0, 0, 0, 0}, cb[4] = {0, 0, 0, 0};
                const int key = n * 8 + a_rin;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int ch = (grp * 2 + kk) * 2 + b_chunk;
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, k_s + key * HD + ((ch ^ (key & 7)) << 4));
                    if constexpr (K4) {
                        if (kk == 0) mma16832_s8(ca, af[0][0], af[0][1], af[0][2], af[0][3], b0, b1);
                        else mma16832_s8(cb, af[1][0], af[1][1], af[1][2], af[1][3], b0, b1);
                    } else {
                        mma16832_s8(ca, af[kk][0], af[kk][1], af[kk][2], af[kk][3], b0, b1);
                    }
                }
                const int ka = n * 8 + 2 * tq;
                if constexpr (K4) {
                    const float k0a = h2f(ks_s[ka * NG + 2 * grp]), k0b = h2f(ks_s[ka * NG + 2 * grp + 1]);
                    const float k1a = h2f(ks_s[(ka + 1) * NG + 2 * grp]), k1b = h2f(ks_s[(ka + 1) * NG + 2 * grp + 1]);
                    s[n][0] = fmaf(qsr[0][grp], k0a * (float) ca[0] + k0b * (float) cb[0], s[n][0]);
                    s[n][1] = fmaf(qsr[0][grp], k1a * (float) ca[1] + k1b * (float) cb[1], s[n][1]);
                    s[n][2] = fmaf(qsr[1][grp], k0a * (float) ca[2] + k0b * (float) cb[2], s[n][2]);
                    s[n][3] = fmaf(qsr[1][grp], k1a * (float) ca[3] + k1b * (float) cb[3], s[n][3]);
                } else {
                    const float ks0 = h2f(ks_s[ka * G4 + grp]), ks1 = h2f(ks_s[(ka + 1) * G4 + grp]);
                    s[n][0] = fmaf(qsr[0][grp] * ks0, (float) ca[0], s[n][0]);
                    s[n][1] = fmaf(qsr[0][grp] * ks1, (float) ca[1], s[n][1]);
                    s[n][2] = fmaf(qsr[1][grp] * ks0, (float) ca[2], s[n][2]);
                    s[n][3] = fmaf(qsr[1][grp] * ks1, (float) ca[3], s[n][3]);
                }
            }
        }

        // mask and online softmax, as the int8 kernel
        const bool full = k0 + I8BC - 1 <= warp_min_qabs;
        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (!full) {
                const int c0 = k0 + n * 8 + 2 * tq, c1 = c0 + 1;
                s[n][0] = c0 <= qabs0 ? s[n][0] : -INFINITY;
                s[n][1] = c1 <= qabs0 ? s[n][1] : -INFINITY;
                s[n][2] = c0 <= qabs1 ? s[n][2] : -INFINITY;
                s[n][3] = c1 <= qabs1 ? s[n][3] : -INFINITY;
            }
            bm0 = fmaxf(bm0, fmaxf(s[n][0], s[n][1]));
            bm1 = fmaxf(bm1, fmaxf(s[n][2], s[n][3]));
        }
        const float nm0 = fmaxf(m0, quad_max(bm0)), nm1 = fmaxf(m1, quad_max(bm1));
        const float ns0 = nm0 == -INFINITY ? 0.0f : nm0 * kScaleLog2;
        const float ns1 = nm1 == -INFINITY ? 0.0f : nm1 * kScaleLog2;
        const float al0 = m0 == -INFINITY ? 0.0f : ex2(m0 * kScaleLog2 - ns0);
        const float al1 = m1 == -INFINITY ? 0.0f : ex2(m1 * kScaleLog2 - ns1);
        m0 = nm0;
        m1 = nm1;
        unsigned pa[KS][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float p00 = ex2(fmaf(s[n][0], kScaleLog2, -ns0)), p01 = ex2(fmaf(s[n][1], kScaleLog2, -ns0));
            const float p10 = ex2(fmaf(s[n][2], kScaleLog2, -ns1)), p11 = ex2(fmaf(s[n][3], kScaleLog2, -ns1));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[n >> 1][(n & 1) * 2] = pack_h2(p00, p01);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_h2(p10, p11);
        }
        l0 = fmaf(l0, al0, bl0);
        l1 = fmaf(l1, al1, bl1);

        // P V: one q4_0 block (32 dims, two 16-dim output blocks) at a time, since its scale is the block's own
        const __half2 origin = __float2half2_rn(1024.0f);
#pragma unroll
        for (int blk = 0; blk < NG; ++blk) {
            __half2 prod[KS][4];
            __half2 mx0 = __float2half2_rn(0.0f), mx1 = mx0;
#pragma unroll
            for (int j = 0; j < KS; ++j) {
                const int key = j * 16 + 2 * tq;
                const __half2 lo = __halves2half2(__ushort_as_half(vs_s[key * NG + blk]), __ushort_as_half(vs_s[(key + 1) * NG + blk]));
                const __half2 hi = __halves2half2(__ushort_as_half(vs_s[(key + 8) * NG + blk]),
                                                  __ushort_as_half(vs_s[(key + 9) * NG + blk]));
                prod[j][0] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][0]), lo);
                prod[j][1] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][1]), lo);
                prod[j][2] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][2]), hi);
                prod[j][3] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][3]), hi);
                mx0 = __hmax2(mx0, __hmax2(prod[j][0], prod[j][2]));
                mx1 = __hmax2(mx1, __hmax2(prod[j][1], prod[j][3]));
            }
            const float rm0 = quad_max(fmaxf(__low2float(mx0), __high2float(mx0)));
            const float rm1 = quad_max(fmaxf(__low2float(mx1), __high2float(mx1)));
            // codes rint(product * inverse), the fp16 inverse rounded down so none passes 255; the step is its reciprocal
            const __half inv0 = __float2half_rd(rm0 > 0.0f ? 255.0f / rm0 : 0.0f);
            const __half inv1 = __float2half_rd(rm1 > 0.0f ? 255.0f / rm1 : 0.0f);
            const float step0 = __hgt(inv0, __float2half(0.0f)) ? 1.0f / __half2float(inv0) : 0.0f;
            const float step1 = __hgt(inv1, __float2half(0.0f)) ? 1.0f / __half2float(inv1) : 0.0f;
            const __half2 i02 = __half2half2(inv0), i12 = __half2half2(inv1);
            unsigned a8[2][4];   // fp16 1024 + n holds n in its low byte
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                unsigned qv[2][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * st + h;
                    const __half2 c0 = __hfma2(prod[j][0], i02, origin), c1 = __hfma2(prod[j][1], i12, origin);
                    const __half2 c2 = __hfma2(prod[j][2], i02, origin), c3 = __hfma2(prod[j][3], i12, origin);
                    qv[h][0] = *reinterpret_cast<const unsigned*>(&c0);
                    qv[h][1] = *reinterpret_cast<const unsigned*>(&c1);
                    qv[h][2] = *reinterpret_cast<const unsigned*>(&c2);
                    qv[h][3] = *reinterpret_cast<const unsigned*>(&c3);
                }
                a8[st][0] = __byte_perm(qv[0][0], qv[0][2], 0x6420);
                a8[st][1] = __byte_perm(qv[0][1], qv[0][3], 0x6420);
                a8[st][2] = __byte_perm(qv[1][0], qv[1][2], 0x6420);
                a8[st][3] = __byte_perm(qv[1][1], qv[1][3], 0x6420);
            }
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int db = blk * 2 + b;
                int ev[4] = {0, 0, 0, 0}, od[4] = {0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    // matrices: cells 32 st + 0..7, 8..15, 16..23, 24..31 of d block db; lane (g, t) gets cells
                    // {2t, 2t+1} at dims {2g, 2g+1}
                    const int key = st * 32 + (a_mat << 3) + a_rin;
                    unsigned r[4];
                    ldsm_x4_t(r[0], r[1], r[2], r[3], v_s + key * HD + ((db ^ (key & 7)) << 4));
                    mma16832_u8s8(ev, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x6420),
                                  __byte_perm(r[2], r[3], 0x6420));
                    mma16832_u8s8(od, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x7531),
                                  __byte_perm(r[2], r[3], 0x7531));
                }
                float (&e)[4] = acc[db][0];
                float (&o)[4] = acc[db][1];
                e[0] = fmaf(e[0], al0, step0 * (float) ev[0]);
                e[1] = fmaf(e[1], al0, step0 * (float) ev[1]);
                e[2] = fmaf(e[2], al1, step1 * (float) ev[2]);
                e[3] = fmaf(e[3], al1, step1 * (float) ev[3]);
                o[0] = fmaf(o[0], al0, step0 * (float) od[0]);
                o[1] = fmaf(o[1], al0, step0 * (float) od[1]);
                o[2] = fmaf(o[2], al1, step1 * (float) od[2]);
                o[3] = fmaf(o[3], al1, step1 * (float) od[3]);
            }
        }
    }

    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const auto store = [&](int row, int e, float m, float l, float inv) {
        if (row >= rows) return;
        const int64_t idx = Split ? ((int64_t) split * T + q0 + row) * n_head + head : (int64_t) (q0 + row) * n_head + head;
        if (Split && tq == 0) part_stats[idx] = make_float2(m, l);
#pragma unroll
        for (int b = 0; b < DB; ++b) {
            const float4 v = make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv);
            if (Split) io_st4(part_rows + idx * HD + b * 16 + 4 * tq, v);   // the splits' rows stay FP32
            else io_st4(out + idx * HD + b * 16 + 4 * tq, v);
        }
    };
    store(row0, 0, m0, l0, i0);
    store(row1, 2, m1, l1, i1);
}

// Block (row, query head), one thread per dim: the splits' rows weighted by sum_s 2^((m_s - M) scale).
template <class IO>
__global__ void __launch_bounds__(HD) prompt_i8_merge_kernel(const float* __restrict__ part_rows,
                                                             const float2* __restrict__ part_stats, int n_head, int T,
                                                             int splits, IO* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        const float2 st = part_stats[((int64_t) s * T + t) * n_head + head];
        if (st.y > 0.0f) mx = fmaxf(mx, st.x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int64_t idx = ((int64_t) s * T + t) * n_head + head;
        const float2 st = part_stats[idx];
        if (st.y > 0.0f) {
            const float w = st.y * exp2f((st.x - mx) * kScaleLog2);
            num = fmaf(w, part_rows[idx * HD + d], num);
            den += w;
        }
    }
    io_st1(out + ((int64_t) t * n_head + head) * HD + d, den > 0.0f ? num / den : 0.0f);
}

// the current device, for a per-device shared-memory opt-in
int opt_in_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        halo::log::error("halo", "gqa attention: device %d out of range", dev);
        std::exit(1);
    }
    return dev;
}

template <int W, bool Split, class IO = float>
void launch_prompt_i8(const IO* q, int n_head, int n_kv, int pos0, int T, IO* out, KvInt8 q8, int splits,
                      float* scratch, cudaStream_t st, KvFar far_kv) {
    static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
    const int dev = opt_in_device();
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(prompt_i8_kernel<W, Split, IO>, cudaFuncAttributeMaxDynamicSharedMemorySize, i8_smem<W>()) !=
            cudaSuccess) {
            halo::log::error("halo", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    float* rows = scratch;
    float2* stats = reinterpret_cast<float2*>(scratch + (size_t) splits * T * n_head * HD);
    const dim3 grid((unsigned) ((T + W * 16 - 1) / (W * 16)), (unsigned) n_head, (unsigned) splits);
    prompt_i8_kernel<W, Split, IO><<<grid, W * 32, i8_smem<W>(), st>>>(q, n_head, n_kv, pos0, T, out, q8, rows, stats, far_kv);
    if (Split) {
        check("gqa_prompt_attn");
        prompt_i8_merge_kernel<IO><<<dim3((unsigned) T, (unsigned) n_head), HD, 0, st>>>(rows, stats, n_head, T, splits, out);
    }
}

// The block caches' prompt launch: the int8 kernel's plan, the same split scratch and merge. K8V4 takes the int8 plan's
// shapes (4 warps unsplit, or 8 warps split or not): its 8-warp shape is 101,376 B of shared memory, exactly sm_89's
// per-block opt-in maximum (with the 1 KiB the driver reserves, the SM's 102,400), one block an SM like int8's 8-warp
// kernel. Q4 runs 4 warps always: its eight K scales a cell put the 8-warp shape 1 KiB past that maximum.
template <int W, bool Split, bool K4, class IO = float>
void launch_prompt_v4(const IO* q, int n_head, int n_kv, int pos0, int T, IO* out, KvK8V4 k8v4, KvQ4 q4, int splits,
                      float* scratch, cudaStream_t st) {
    static std::atomic<bool> raised[64];
    const int dev = opt_in_device();
    if (!raised[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(prompt_v4_kernel<W, Split, K4, IO>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 v4_smem<W, K4>()) != cudaSuccess) {
            halo::log::error("halo", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        raised[dev].store(true, std::memory_order_release);
    }
    float* rows = scratch;
    float2* stats = reinterpret_cast<float2*>(scratch + (size_t) splits * T * n_head * HD);
    const dim3 grid((unsigned) ((T + W * 16 - 1) / (W * 16)), (unsigned) n_head, (unsigned) splits);
    prompt_v4_kernel<W, Split, K4, IO><<<grid, W * 32, v4_smem<W, K4>(), st>>>(q, n_head, n_kv, pos0, T, out, k8v4, q4, rows, stats);
    if (Split) {
        check("gqa_prompt_attn");
        prompt_i8_merge_kernel<IO><<<dim3((unsigned) T, (unsigned) n_head), HD, 0, st>>>(rows, stats, n_head, T, splits, out);
    }
}

// The int8 launch's block shape and key splits, as Infernix plans them (int8_fast_prompt_plan): a launch takes about
// (waves) x (visible cells / splits) sweeps of one cell by an 8-warp block; a 4-warp block covers half the rows in 0.8 of
// that time, unsplit; a split launch also writes and merges every split's rows, about half a cell's sweep per row, split
// and head over 24 heads.  Infernix measured this within 0.14 % of the fastest choice on average, 3.8 % at worst.
struct I8Plan {
    int warps = 8, splits = 1;
};
std::atomic<int> plan_device{-1};   // gqa_prompt_plan_device
I8Plan plan_prompt_i8(int n_head, int64_t T, int64_t cells, size_t split_floats) {
    constexpr int kMaxSplits = 8;
    int dev = plan_device.load(std::memory_order_relaxed), sms = 0;
    if (dev < 0) cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    sms = std::max(sms, 1);
    const auto waves = [&](int64_t blocks) { return (double) ((blocks + sms - 1) / sms); };
    const double keys = (double) cells;
    I8Plan best{4, 1};
    double best_cost = waves((T + 63) / 64 * n_head) * 0.8 * keys;
    const int64_t blocks = (T + 127) / 128 * n_head;
    for (int s = 1; s <= kMaxSplits; ++s) {
        if (s > 1 && gqa_prompt_split_floats(T, n_head, s) > split_floats) break;
        double cost = waves(blocks * s) * keys / s;
        if (s > 1) cost += 0.5 / 24.0 * (double) T * s * n_head;
        if (cost < best_cost) {
            best_cost = cost;
            best = {8, s};
        }
    }
    return best;
}

// A decode window's (or a batch window's segments') launch: the argument checks, the ring's chunk base, then one
// launch_segs.  `path` picks the kernel: the engine's own choice, or the 16-row-tile / window kernel the parity tests and
// the bench force to compare against it.
void decode_attn(DecPath path, const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                 int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream, const int32_t* pos_dev,
                 KvInt8 q8, const float* gate_q, uint16_t* gated16, KvFar far_kv, KvK8V4 k8v4, KvQ4 q4, int64_t window,
                 int64_t cap) {
    const bool k4_ring = (k8v4.k || q4.k) && cap > 0;
    const int formats = (q8.k != nullptr) + (k8v4.k != nullptr) + (q4.k != nullptr);
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || pos0 + T > max_cells ||
        window < 0 || cap < 0 || (cap > 0 && (window <= 0 || window + T - 1 > cap)) || formats > 1 ||
        ((k8v4.k || q4.k) && far_kv.k) || k4_ring) {
        halo::log::error("halo", "gqa_decode_attn: invalid arguments%s",
                          k4_ring ? " (a 4-bit cache is never a ring)" : "");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.cap = cap > 0 ? cap : max_cells;
    r.window = window;
    r.pos_dev = pos_dev;
    // A ring holds positions [pos0 - window + 1, pos0 + T) only, so its sweep starts at the window's first chunk, as
    // gqa_window_attn's does: from chunk 0 a ring of `cap` cells reaches position cap and no further, and past it every
    // row read nothing (sum 0, its output NaN). One chunk more than the ring's covers the window's offset in its first
    // chunk (window + T - 1 <= cap, plus at most CHUNK - 1 cells before lo). The base comes from the host's pos0: a ring's
    // caller (the drafter's window, mtp_window) uploads pos0 on every call and is never captured in a graph.
    int n_chunks = (int) ((r.cap + CHUNK - 1) / CHUNK);
    if (cap > 0) {
        r.base = (int) (std::max<int64_t>(0, pos0 - window + 1) / CHUNK);
        n_chunks += 1;
    }
    launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, path, q8, gate_q,
                       gated16, far_kv, k8v4, q4);
}

void decode_segs(DecPath path, const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                 int64_t max_cells, float* out, void* stream, const float* gate_q, uint16_t* gated16) {
    if (n_seg < 1 || n_seg > kMaxSegs || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX) {
        halo::log::error("halo", "gqa_decode_attn_segs: invalid arguments");
        std::exit(1);
    }
    DecSegs d;
    d.n = n_seg;
    for (int k = 0; k < n_seg; ++k) {
        const GqaSeg& g = segs[k];
        if (g.T < 1 || g.pos0 + g.T > max_cells || d.row0[k] + g.T > kGqaMaxWindow ||
            (((g.k8v4.k != nullptr) + (g.q4.k != nullptr) + (g.q8.k != nullptr)) > 1) ||
            ((g.k8v4.k || g.q4.k) && g.far_kv.k) ||
            ((g.q4.k != nullptr) != (segs[0].q4.k != nullptr))) {   // launch_segs picks the Q4 kernels from segment 0
            halo::log::error("halo", "gqa_decode_attn_segs: invalid segment");
            std::exit(1);
        }
        d.row0[k + 1] = d.row0[k] + g.T;
        d.kc[k] = (const __half*) g.k_cache;
        d.vc[k] = (const __half*) g.v_cache;
        d.q8[k] = g.q8;
        d.far_kv[k] = g.far_kv;
        d.k8v4[k] = g.k8v4;
        d.q4[k] = g.q4;
        d.r[k].pos0 = g.pos0;
        d.r[k].cap = max_cells;
        d.r[k].pos_dev = g.pos_dev;
    }
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    launch_segs<256>(q, n_head, n_kv, d, scratch, n_chunks, out, (cudaStream_t) stream, path, gate_q, gated16);
}

}  // namespace

size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim) {
    const int64_t chunks = (max_cells + CHUNK - 1) / CHUNK + 1;
    return (size_t) T * n_head * decode_splits(chunks) * (head_dim + 2);
}

void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                     const int32_t* pos_dev, KvInt8 q8, const float* gate_q, uint16_t* gated16, KvFar far_kv,
                     KvK8V4 k8v4, KvQ4 q4, int64_t window, int64_t cap) {
    decode_attn(DecPath::kEngine, q, k_cache, v_cache, n_head, n_kv, pos0, T, scratch, max_cells, out, stream, pos_dev, q8,
                gate_q, gated16, far_kv, k8v4, q4, window, cap);
}

void gqa_decode_attn_segs(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                          int64_t max_cells, float* out, void* stream, const float* gate_q, uint16_t* gated16) {
    decode_segs(DecPath::kEngine, q, n_head, n_kv, segs, n_seg, scratch, max_cells, out, stream, gate_q, gated16);
}

namespace test {

// tests/kernels/gqa_decode_ref.hpp: the same launches on the kernels the engine does not select (the 16-row-tile launch
// and, for K8V4, the window kernel), for the parity tests and the bench only.
void gqa_decode_attn_tiled(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                          int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                          const int32_t* pos_dev, KvInt8 q8, const float* gate_q, uint16_t* gated16, KvFar far_kv,
                          KvK8V4 k8v4, int64_t window, int64_t cap) {
    decode_attn(DecPath::kTiled, q, k_cache, v_cache, n_head, n_kv, pos0, T, scratch, max_cells, out, stream, pos_dev, q8,
                gate_q, gated16, far_kv, k8v4, {}, window, cap);
}

void gqa_decode_attn_segs_tiled(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                               int64_t max_cells, float* out, void* stream, const float* gate_q, uint16_t* gated16) {
    decode_segs(DecPath::kTiled, q, n_head, n_kv, segs, n_seg, scratch, max_cells, out, stream, gate_q, gated16);
}

void gqa_decode_attn_window(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                           int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                           const int32_t* pos_dev, KvInt8 q8, const float* gate_q, uint16_t* gated16, KvFar far_kv,
                           KvK8V4 k8v4, int64_t window, int64_t cap) {
    decode_attn(DecPath::kWindow, q, k_cache, v_cache, n_head, n_kv, pos0, T, scratch, max_cells, out, stream, pos_dev, q8,
                gate_q, gated16, far_kv, k8v4, {}, window, cap);
}

void gqa_decode_attn_segs_window(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                                int64_t max_cells, float* out, void* stream, const float* gate_q, uint16_t* gated16) {
    decode_segs(DecPath::kWindow, q, n_head, n_kv, segs, n_seg, scratch, max_cells, out, stream, gate_q, gated16);
}

}  // namespace test

void gqa_window_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv, int head_dim,
                     int64_t pos0, int T, int64_t hi, int64_t window, int64_t cap, float* scratch, int64_t scratch_cells,
                     float* out, void* stream) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || window <= 0 ||
        (head_dim != 128 && head_dim != 256) || hi < pos0 + T) {
        halo::log::error("halo", "gqa_window_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.hi = hi;
    r.window = window;
    r.cap = cap;
    r.causal = 0;
    const int64_t lo = std::max<int64_t>(0, pos0 - window + 1);
    r.base = (int) (lo / CHUNK);
    const int used = (int) ((hi + CHUNK - 1) / CHUNK) - r.base;
    const int n_chunks = (int) ((scratch_cells + CHUNK - 1) / CHUNK) + 1;
    if (used > n_chunks) {
        halo::log::error("halo", "gqa_window_attn: the window passes its scratch");
        std::exit(1);
    }
    if (head_dim == 128) launch_window<128>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, DecPath::kEngine);
    else launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, DecPath::kEngine);
}

void gqa_prompt_plan_device(int device) { plan_device.store(device, std::memory_order_relaxed); }

size_t gqa_prompt_split_floats(int64_t T, int n_head, int splits) {
    return (size_t) splits * T * n_head * (HD + 2);   // the splits' rows, then their (max, sum) pairs
}

void gqa_prompt_plan_query(int n_head, int64_t T, int64_t cells, size_t split_floats, int* warps, int* splits) {
    const I8Plan p = plan_prompt_i8(n_head, T, cells, split_floats);
    if (warps) *warps = p.warps;
    if (splits) *splits = p.splits;
}

void gqa_prompt_attn_bf16(const uint16_t* q, int n_head, int n_kv, int64_t pos0, int64_t T, uint16_t* out, void* stream,
                          KvInt8 q8, float* split_scratch, size_t split_floats, KvFar far_kv, KvK8V4 k8v4, KvQ4 q4) {
    if (T <= 0) return;
    const int formats = (q8.k != nullptr) + (k8v4.k != nullptr) + (q4.k != nullptr);
    if (formats != 1 || ((k8v4.k || q4.k) && far_kv.k) || n_kv < 1 || n_head % n_kv || pos0 + T > INT32_MAX) {
        halo::log::error("halo", "gqa_prompt_attn_bf16: one int8 or block cache and a supported head layout");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
    if (q4.k) {
        if (p.splits == 1) launch_prompt_v4<4, false, true>(qb, n_head, n_kv, (int) pos0, (int) T, ob, k8v4, q4, 1, nullptr, st);
        else launch_prompt_v4<4, true, true>(qb, n_head, n_kv, (int) pos0, (int) T, ob, k8v4, q4, p.splits, split_scratch, st);
    } else if (k8v4.k) {
        if (p.warps == 4) launch_prompt_v4<4, false, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, k8v4, q4, 1, nullptr, st);
        else if (p.splits == 1) launch_prompt_v4<8, false, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, k8v4, q4, 1, nullptr, st);
        else launch_prompt_v4<8, true, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, k8v4, q4, p.splits, split_scratch, st);
    } else if (p.warps == 4) launch_prompt_i8<4, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st, far_kv);
    else if (p.splits == 1) launch_prompt_i8<8, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st, far_kv);
    else launch_prompt_i8<8, true>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, p.splits, split_scratch, st, far_kv);
    check("gqa_prompt_attn_bf16");
}

void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8, float* split_scratch, size_t split_floats,
                     KvFar far_kv, KvK8V4 k8v4, KvQ4 q4) {
    if (T <= 0) return;
    const int formats = (q8.k != nullptr) + (k8v4.k != nullptr) + (q4.k != nullptr);
    if (n_kv < 1 || n_head % n_kv || formats > 1 || (!q8.k && !k8v4.k && !q4.k && PROWS % (n_head / n_kv)) ||
        ((k8v4.k || q4.k) && far_kv.k) || pos0 + T > INT32_MAX) {
        halo::log::error("halo", "gqa_prompt_attn: unsupported head layout or cell count");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (q4.k) {
        const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
        if (p.splits == 1) launch_prompt_v4<4, false, true>(q, n_head, n_kv, (int) pos0, (int) T, out, k8v4, q4, 1, nullptr, st);
        else launch_prompt_v4<4, true, true>(q, n_head, n_kv, (int) pos0, (int) T, out, k8v4, q4, p.splits, split_scratch, st);
    } else if (k8v4.k) {
        const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
        if (p.warps == 4) launch_prompt_v4<4, false, false>(q, n_head, n_kv, (int) pos0, (int) T, out, k8v4, q4, 1, nullptr, st);
        else if (p.splits == 1) launch_prompt_v4<8, false, false>(q, n_head, n_kv, (int) pos0, (int) T, out, k8v4, q4, 1, nullptr, st);
        else launch_prompt_v4<8, true, false>(q, n_head, n_kv, (int) pos0, (int) T, out, k8v4, q4, p.splits, split_scratch, st);
    } else if (q8.k) {
        const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
        if (p.warps == 4) launch_prompt_i8<4, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st, far_kv);
        else if (p.splits == 1) launch_prompt_i8<8, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st, far_kv);
        else launch_prompt_i8<8, true>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, p.splits, split_scratch, st, far_kv);
    } else {
        static std::atomic<bool> raised[64];   // the shared-memory opt-in is per device
        const int dev = opt_in_device();
        if (!raised[dev].load(std::memory_order_acquire)) {
            if (cudaFuncSetAttribute(prompt_f16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PromptSmem16)) !=
                cudaSuccess) {
                halo::log::error("halo", "gqa_prompt_attn: shared-memory opt-in failed");
                std::exit(1);
            }
            raised[dev].store(true, std::memory_order_release);
        }
        const int pb = PROWS / (n_head / n_kv);
        const dim3 grid((unsigned) ((T + pb - 1) / pb), (unsigned) n_kv);
        prompt_f16_kernel<<<grid, PNW * 32, sizeof(PromptSmem16), st>>>(q, (const __half*) k_cache, (const __half*) v_cache,
                                                                         n_head, n_kv, pos0, T, out, far_kv);
    }
    check("gqa_prompt_attn");
}

}  // namespace halo::kernels
