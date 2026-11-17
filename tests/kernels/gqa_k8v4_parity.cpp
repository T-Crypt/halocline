// tests/kernels/gqa_k8v4_parity.cpp - the dense path's K8V4 cache (halo/kernels/gqa_attn.hpp: K in the int8 form, V in
// rotated q4_0 blocks) against a double-precision CPU reference, at the Qwen3.5-27B attention geometry (24 query heads,
// 4 KV heads, 256 wide, 64 rotated dims, M-RoPE sections 11/11/10).
//
// The cache write takes a row through the RMS norm (rsqrtf), the rope (sincosf) and then the int8 rule, so a host
// reference cannot reproduce its bits: rsqrtf and sincosf are the device's own.  What a reference can do is prove the
// rule: the K half is byte-for-byte the int8 cache's write of the same launch (the same y, the same statements), it is
// the host's norm + rope + int8 quantization to within one step, and V's blocks are bitwise the host quantizer over the
// host-rotated row, since V is neither normed nor roped.
//
// Checks:
//   1. attn_qk_prepare's K8V4 write: K's codes and per-64 scales are the int8 cache's write bitwise, that write is the
//      host's norm + rope + int8 rule within a step, V's blocks are bitwise the host rule over the rotated row;
//   2. the V rotation inside the write is fwht256_cuda's bits (the same row through the shared kernel);
//   3. gqa_decode_attn + the output's inverse rotation against the fp64 reference on the cache's dequantized cells, over
//      every row of one window; and one row's bits are the same whatever the window width;
//   4. gqa_prompt_attn (4-warp, and split over three key ranges) + the inverse rotation, at rows on and around the tile
//      and query-block edges, against the same reference;
//   5. gqa_prompt_attn_bf16's rows are the FP32 launch's on the same (bf16-exact) queries, rounded to bf16 once: bitwise.
// Exit 77 without a GPU.  GPU test: build it, do not run it while a production engine holds the card.
#include "halo/kernels/bf16_bits.hpp"
#include "halo/kernels/dense_ops.hpp"
#include "halo/kernels/f16_bits.hpp"
#include "halo/kernels/gqa_attn.hpp"
#include "halo/kernels/kv_q4_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = halo::kernels;

namespace {
constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;
constexpr double kPi = 3.14159265358979323846;
int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class T> T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, (n * sizeof(T) + 255) & ~(size_t) 255), "cudaMalloc");
    ck(cudaMemset(p, 0, (n * sizeof(T) + 255) & ~(size_t) 255), "cudaMemset");
    return (T*) p;
}
float h2f(uint16_t h) { return k::f32_from_f16(h); }

// The orthonormal 256-point Walsh-Hadamard transform, the reference's own (stages 1..128, then 1/16).
void fwht_host(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < n; i += 2 * s)
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j], v = x[i + j + s];
                x[i + j] = u + v;
                x[i + j + s] = u - v;
            }
    for (int i = 0; i < n; ++i) x[i] *= 1.0f / 16.0f;
}

// out[h][d] over `visible` cells of a row: scores on the fp64 dequantized K, the mix on the (un-rotated) V
void ref_attn(const float* q, const std::vector<float>& kdq, const std::vector<float>& vdq, int64_t visible, double* out) {
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        std::vector<double> s((size_t) visible);
        double mx = -1e300;
        for (int64_t c = 0; c < visible; ++c) {
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[h * HD + i] * kdq[((size_t) c * NKV + kvh) * HD + i];
            s[(size_t) c] = d / 16.0;
            mx = std::max(mx, s[(size_t) c]);
        }
        double l = 0;
        for (auto& x : s) { x = std::exp(x - mx); l += x; }
        for (int i = 0; i < HD; ++i) {
            double a = 0;
            for (int64_t c = 0; c < visible; ++c) a += s[(size_t) c] * vdq[((size_t) c * NKV + kvh) * HD + i];
            out[h * HD + i] = a / l;
        }
    }
}

struct Err {
    double abs = 0, rel = 0;
    void add(double got, double want) {
        abs = std::max(abs, std::fabs(got - want));
        rel = std::max(rel, std::fabs(got - want) / std::max(std::fabs(want), 1e-2));
    }
};

void report(bool ok, const char* what) {
    std::printf("%s: %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

// the first byte two blocks differ at (-1: none), to name the region and the cell a failure starts at
template <class T> int64_t first_diff(const T* a, const T* b, size_t n) {
    const uint8_t* x = reinterpret_cast<const uint8_t*>(a);
    const uint8_t* y = reinterpret_cast<const uint8_t*>(b);
    for (size_t i = 0; i < n * sizeof(T); ++i)
        if (x[i] != y[i]) return (int64_t) i;
    return -1;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    const int64_t L = 1200, max_cells = 2048;   // cells written, capacity
    const int64_t cell0 = 37;                   // the write lands at 37 .. 37 + L - 1
    std::vector<float> K((size_t) L * NKV * HD), V(K.size());
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    k::RopeSpec rope;
    rope.rot = ROT;
    rope.theta = 1e7f;
    for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
    const float eps = 1e-6f;

    // ---- the host reference for V: rotated by the reference transform, then q4_0 (halo/kernels/kv_q4_layout.hpp)
    std::vector<float> vrot(V);
    for (int64_t row = 0; row < L * NKV; ++row) fwht_host(&vrot[(size_t) row * HD]);
    std::vector<k::KvQ4Block> rv4((size_t) L * NKV * (HD / k::kKvQ4Block));
    for (int64_t row = 0; row < L * NKV; ++row)
        for (int b = 0; b < HD / k::kKvQ4Block; ++b)
            rv4[(size_t) row * (HD / k::kKvQ4Block) + b] = k::kv_q4_quantize_host(&vrot[(size_t) row * HD + b * k::kKvQ4Block]);

    // ---- device caches, written by attn_qk_prepare on one set of inputs: the K8V4 form and, from the same launch
    // parameters, the int8 one whose K half the K8V4 form must be
    int8_t* d_k = dev<int8_t>((size_t) max_cells * NKV * HD);
    uint16_t* d_ks = dev<uint16_t>((size_t) max_cells * NKV * (HD / 64));
    uint8_t* d_vq = dev<uint8_t>((size_t) max_cells * NKV * k::kv_q4_head_bytes(HD));
    int8_t* d_ik = dev<int8_t>((size_t) max_cells * NKV * HD);
    int8_t* d_iv = dev<int8_t>((size_t) max_cells * NKV * HD);
    uint16_t* d_iks = dev<uint16_t>((size_t) max_cells * NKV * (HD / 64));
    uint16_t* d_ivs = dev<uint16_t>((size_t) max_cells * NKV * (HD / 64));
    float *d_kf = dev<float>(K.size()), *d_vf = dev<float>(V.size()), *d_qo = dev<float>((size_t) L * NH * HD);
    k::KvK8V4 cache{d_k, d_ks, d_vq};
    k::KvInt8 cache8{d_ik, d_iv, d_iks, d_ivs};
    {
        std::vector<float> qf((size_t) L * NH * 2 * HD), qn(HD, 1.0f), kn(HD, 1.0f);
        for (auto& x : qf) x = nd(rng);
        std::vector<int32_t> pos((size_t) L * 3);
        for (int64_t t = 0; t < L; ++t) pos[(size_t) t * 3] = pos[(size_t) t * 3 + 1] = pos[(size_t) t * 3 + 2] = (int32_t) (cell0 + t);
        float* d_qf = dev<float>(qf.size());
        float *d_qn = dev<float>(HD), *d_kn = dev<float>(HD);
        int32_t* d_pos = dev<int32_t>(pos.size());
        ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "q");
        ck(cudaMemcpy(d_qn, qn.data(), HD * 4, cudaMemcpyHostToDevice), "qn");
        ck(cudaMemcpy(d_kn, kn.data(), HD * 4, cudaMemcpyHostToDevice), "kn");
        ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "pos");
        ck(cudaMemcpy(d_kf, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
        ck(cudaMemcpy(d_vf, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        k::attn_qk_prepare(d_qf, d_kf, d_vf, d_qn, d_kn, eps, rope, d_pos, L, NH, NKV, d_qo, nullptr, nullptr, cell0, nullptr, 256,
                           true, 0, nullptr, k::KvInt8{}, k::KvFar{}, cache);
        k::attn_qk_prepare(d_qf, d_kf, d_vf, d_qn, d_kn, eps, rope, d_pos, L, NH, NKV, d_qo, nullptr, nullptr, cell0, nullptr, 256,
                           true, 0, nullptr, cache8, k::KvFar{}, k::KvK8V4{});
        ck(cudaDeviceSynchronize(), "prepare");
        cudaFree(d_qf); cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_pos);
    }
    // [1] the written bytes, cell by cell, region by region
    std::vector<int8_t> hk((size_t) L * NKV * HD), ik((size_t) L * NKV * HD);
    std::vector<uint16_t> hks((size_t) L * NKV * (HD / 64)), iks(hks.size());
    std::vector<uint8_t> hv((size_t) L * NKV * k::kv_q4_head_bytes(HD));
    ck(cudaMemcpy(hk.data(), d_k + (size_t) cell0 * NKV * HD, hk.size(), cudaMemcpyDeviceToHost), "k");
    ck(cudaMemcpy(hks.data(), d_ks + (size_t) cell0 * NKV * (HD / 64), hks.size() * 2, cudaMemcpyDeviceToHost), "ks");
    ck(cudaMemcpy(hv.data(), d_vq + (size_t) cell0 * NKV * k::kv_q4_head_bytes(HD), hv.size(), cudaMemcpyDeviceToHost), "v");
    ck(cudaMemcpy(ik.data(), d_ik + (size_t) cell0 * NKV * HD, ik.size(), cudaMemcpyDeviceToHost), "int8 k");
    ck(cudaMemcpy(iks.data(), d_iks + (size_t) cell0 * NKV * (HD / 64), iks.size() * 2, cudaMemcpyDeviceToHost), "int8 ks");
    {
        size_t bad_k = 0, bad_ks = 0, bad_v = 0;
        for (size_t i = 0; i < hk.size(); ++i) bad_k += hk[i] != ik[i];
        for (size_t i = 0; i < hks.size(); ++i) bad_ks += hks[i] != iks[i];
        for (size_t i = 0; i < hv.size(); ++i) bad_v += hv[i] != reinterpret_cast<const uint8_t*>(rv4.data())[i];
        const int64_t at_k = first_diff(hk.data(), ik.data(), hk.size());
        const int64_t at_ks = first_diff(hks.data(), iks.data(), hks.size());
        const int64_t at_v = first_diff(hv.data(), reinterpret_cast<const uint8_t*>(rv4.data()), hv.size());
        std::printf("  K8V4 write, K half vs the int8 cache's write: %zu of %zu codes (%.2f%%) and %zu of %zu scales differ\n", bad_k,
                    hk.size(), 100.0 * (double) bad_k / (double) hk.size(), bad_ks, hks.size());
        if (at_k >= 0) {
            const int64_t row = at_k / HD;
            std::printf("    first K code at byte %lld: cell %lld, kv head %lld, dim %lld\n", (long long) at_k,
                        (long long) (cell0 + row / NKV), (long long) (row % NKV), (long long) (at_k % HD));
        }
        if (at_ks >= 0)
            std::printf("    first K scale at byte %lld: row %lld (cell %lld, kv head %lld), group %lld\n", (long long) at_ks,
                        (long long) (at_ks / 2 / (HD / 64)), (long long) (cell0 + at_ks / 2 / (HD / 64) / NKV),
                        (long long) (at_ks / 2 / (HD / 64) % NKV), (long long) (at_ks / 2 % (HD / 64)));
        report(bad_k == 0 && bad_ks == 0, "attn_qk_prepare's K8V4 K half is the int8 cache's write bitwise");
        std::printf("  K8V4 write, V blocks vs the host rotate + q4_0 rule: %zu bytes of %zu differ\n", bad_v, hv.size());
        if (at_v >= 0)
            std::printf("    first V byte at %lld: cell %lld, kv head %lld, block %lld, offset %lld\n", (long long) at_v,
                        (long long) (cell0 + at_v / (int64_t) k::kv_q4_head_bytes(HD) / NKV),
                        (long long) (at_v / (int64_t) k::kv_q4_head_bytes(HD) % NKV),
                        (long long) (at_v % (int64_t) k::kv_q4_head_bytes(HD) / 18),
                        (long long) (at_v % (int64_t) k::kv_q4_head_bytes(HD) % 18));
        // the device quantizer's 1 / d is the fast-math reciprocal, so a value that sits on a code boundary can round one
        // step away from the host's: scales must be the same bits, a code at most one step off, and such ties rare
        size_t scale_off = 0, code_tie = 0, code_far = 0;
        const auto* ref_blk = rv4.data();
        const auto* dev_blk = reinterpret_cast<const k::KvQ4Block*>(hv.data());
        for (size_t b = 0; b < rv4.size(); ++b) {
            scale_off += dev_blk[b].d != ref_blk[b].d;
            for (int i = 0; i < 16; ++i)
                for (int sh = 0; sh <= 4; sh += 4) {
                    const int a = (dev_blk[b].qs[i] >> sh) & 15, r = (ref_blk[b].qs[i] >> sh) & 15;
                    const int step = a > r ? a - r : r - a;
                    code_tie += step == 1;
                    code_far += step > 1;
                }
        }
        const size_t n_codes = rv4.size() * 32;
        std::printf("  K8V4 write, V blocks: %zu of %zu scales differ; %zu codes one step off (ties), %zu further, of %zu\n",
                    scale_off, rv4.size(), code_tie, code_far, n_codes);
        report(scale_off == 0 && code_far == 0 && code_tie * 100000 <= n_codes,
               "attn_qk_prepare's K8V4 V blocks are the host rotate + q4_0 rule (scales bitwise, codes to a rounding tie)");
    }
    // [1b] the K half against the host's own rule, with the one loose end the write cannot avoid: the row goes through
    // rsqrtf and the rope through sincosf, so the host's values differ in their last bits and a code may sit one step
    // away.  A wrong layout, stride, group or cell is thousands of steps out and fails here.
    {
        std::vector<float> inv_freq(ROT / 2);
        for (int i = 0; i < ROT / 2; ++i) inv_freq[(size_t) i] = (float) std::pow((double) rope.theta, -2.0 * (double) i / (double) ROT);
        int64_t codes_off = 0, worst = 0, scales_off = 0;
        double worst_scale = 0;
        for (int64_t t = 0; t < L; ++t)
            for (int h = 0; h < NKV; ++h) {
                const int64_t row = t * NKV + h;
                const float* x = &K[(size_t) row * HD];
                double ss = 0;
                for (int d = 0; d < HD; ++d) ss += (double) x[d] * x[d];
                const double inv = 1.0 / std::sqrt(ss / (double) HD + (double) eps);
                double y[HD];
                for (int d = 0; d < HD; ++d) y[d] = (double) x[d] * inv;   // k_norm is 1 in this test
                for (int i = 0; i < ROT / 2; ++i) {
                    const float ang = (float) ((float) (cell0 + t) * inv_freq[(size_t) i]);
                    const double c = std::cos((double) ang), sn = std::sin((double) ang);
                    const double u = y[i], v = y[i + ROT / 2];
                    y[i] = u * c - v * sn;
                    y[i + ROT / 2] = v * c + u * sn;
                }
                for (int g = 0; g < HD / 64; ++g) {
                    const float sd = h2f(hks[(size_t) row * (HD / 64) + g]);
                    float amax = 0.0f;
                    for (int e = 0; e < 64; ++e) amax = std::max(amax, (float) std::fabs(y[g * 64 + e]));
                    const float sh = h2f(k::f16_from_f32(amax / 127.0f));
                    if (sh > 0.0f && std::fabs(sd / sh - 1.0f) > 1.0e-3f) {
                        ++scales_off;
                        worst_scale = std::max(worst_scale, (double) std::fabs(sd / sh - 1.0f));
                    }
                    if (!(sd > 0.0f)) continue;
                    for (int e = 0; e < 64; ++e) {
                        const int want = std::max(-127, std::min(127, (int) std::nearbyint(y[g * 64 + e] / (double) sd)));
                        const int64_t steps = std::llabs((long long) want - (long long) hk[(size_t) row * HD + g * 64 + e]);
                        if (steps > 1) ++codes_off;
                        worst = std::max(worst, steps);
                    }
                }
            }
        std::printf("  K8V4 write, K vs the host norm + rope + int8 rule: worst code %lld step(s) off, %lld codes over one, "
                    "%lld scales over 1e-3 (worst %.2g)\n", (long long) worst, (long long) codes_off, (long long) scales_off,
                    worst_scale);
        report(codes_off == 0 && scales_off == 0, "attn_qk_prepare's K8V4 K is the host norm + rope + int8 rule within a step");
    }
    // [2] the write's V rotation is fwht256_cuda's bits
    {
        const size_t rows = (size_t) L * NKV;
        float* d_a = dev<float>(rows * HD);
        float* d_b = dev<float>(rows * HD);
        ck(cudaMemcpy(d_a, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        k::fwht256_cuda(d_a, d_b, (int64_t) rows, nullptr);
        ck(cudaDeviceSynchronize(), "fwht");
        std::vector<float> hb(rows * HD);
        ck(cudaMemcpy(hb.data(), d_b, hb.size() * 4, cudaMemcpyDeviceToHost), "fwht out");
        double worst = 0;
        for (size_t i = 0; i < hb.size(); ++i) worst = std::max(worst, (double) std::fabs(hb[i] - vrot[i]));
        std::printf("  fwht256_cuda vs the host transform: max abs %.3g\n", worst);
        report(worst <= 1e-5, "the shared rotation is the reference's transform");
        cudaFree(d_a);
        cudaFree(d_b);
    }

    // ---- the cache's cells as the readers dequantize them, at their own positions: cells 0 .. cell0 - 1 are the
    // zeroed cache, K is a code times its group's scale and V a code times its block's, each rounded to fp16 as the decode
    // kernel's tiles are; V comes back un-rotated here (the kernels read it rotated and the engine rotates the output)
    std::vector<float> kdq((size_t) (cell0 + L) * NKV * HD, 0.0f), vdq(kdq.size(), 0.0f);
    for (int64_t row = 0; row < L * NKV; ++row) {
        const size_t dst = (size_t) ((cell0 + row / NKV) * NKV + row % NKV) * HD;
        for (int d = 0; d < HD; ++d) {
            kdq[dst + d] = h2f(k::f16_from_f32((float) hk[(size_t) row * HD + d] * h2f(hks[(size_t) row * (HD / 64) + d / 64])));
            int codes[8];
            k::kv_q4_codes8(rv4[(size_t) row * 8 + d / 32].qs, (d & 31) & ~7, codes);
            vdq[dst + d] = h2f(k::f16_from_f32(k::kv_q4_dequant(codes[d & 7], h2f(rv4[(size_t) row * 8 + d / 32].d))));
        }
    }
    std::vector<float> vtrue(vdq);
    for (int64_t row = 0; row < (cell0 + L) * NKV; ++row) fwht_host(&vtrue[(size_t) row * HD]);   // every cell's V row

    // ---- [3] decode windows over the K8V4 cache, the output rotated back once per head; every width ends on the same
    // row, so that row's bits must not move, and the widest window's rows are the reference's
    std::vector<float> q((size_t) k::kGqaMaxWindow * NH * HD);
    for (auto& x : q) x = 2.0f * nd(rng);
    float* d_q = dev<float>(q.size());
    float* d_out = dev<float>(q.size());
    float* d_scr = dev<float>(k::gqa_decode_scratch_floats(max_cells, k::kGqaMaxWindow, NH));
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
    {
        const int64_t last_cell = cell0 + L - 1;   // row kGqaMaxWindow - 1 of the array, in every launch
        std::vector<double> want((size_t) NH * HD);
        Err ed;
        bool bitwise = true, have = false;
        std::vector<float> first((size_t) NH * HD), got((size_t) k::kGqaMaxWindow * NH * HD);
        for (int T = 1; T <= k::kGqaMaxWindow; ++T) {
            const int64_t p0 = last_cell - T + 1;
            const float* qw = &q[(size_t) (k::kGqaMaxWindow - T) * NH * HD];
            k::gqa_decode_attn(qw, nullptr, nullptr, NH, NKV, p0, T, d_scr, max_cells, d_out, nullptr, nullptr, k::KvInt8{},
                               nullptr, nullptr, k::KvFar{}, cache);
            k::fwht256_inplace_cuda(d_out, (int64_t) T * NH, nullptr);   // the engine's inverse rotation of each head
            ck(cudaDeviceSynchronize(), "decode");
            ck(cudaMemcpy(got.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "decode out");
            const float* last = &got[(size_t) (T - 1) * NH * HD];
            if (!have) { std::memcpy(first.data(), last, first.size() * 4); have = true; }
            else if (std::memcmp(first.data(), last, first.size() * 4) != 0) bitwise = false;
            if (T == k::kGqaMaxWindow)   // every row of one window, each at its own position
                for (int t = 0; t < T; ++t) {
                    ref_attn(&q[(size_t) t * NH * HD], kdq, vtrue, p0 + t + 1, want.data());
                    for (int j = 0; j < NH * HD; ++j) ed.add(got[(size_t) t * NH * HD + j], want[(size_t) j]);
                }
        }
        std::printf("  K8V4 decode: max abs %.3g rel %.3g vs double over %d rows; the last row bitwise equal across widths "
                    "1-%d: %s\n", ed.abs, ed.rel, k::kGqaMaxWindow, k::kGqaMaxWindow, bitwise ? "yes" : "NO");
        report(ed.abs < 2e-3 && bitwise, "gqa_decode_attn over a K8V4 cache matches the fp64 reference");
    }

    // ---- [4] prompt chunks: 300 rows at the end of the cache, plain and split over three key ranges
    {
        const int T = 300;
        const int64_t pp = cell0 + L - T;
        std::vector<float> qp((size_t) T * NH * HD);
        for (auto& x : qp) x = 2.0f * nd(rng);
        float *d_qp = dev<float>(qp.size()), *d_op = dev<float>(qp.size());
        const size_t sf = k::gqa_prompt_split_floats(T, NH, 3);
        float* d_split = dev<float>(sf);
        ck(cudaMemcpy(d_qp, qp.data(), qp.size() * 4, cudaMemcpyHostToDevice), "qp");
        for (int cfg = 0; cfg < 2; ++cfg) {
            k::gqa_prompt_attn(d_qp, nullptr, nullptr, NH, NKV, pp, T, d_op, nullptr, k::KvInt8{}, cfg ? d_split : nullptr,
                               cfg ? sf : 0, k::KvFar{}, cache);
            k::fwht256_inplace_cuda(d_op, (int64_t) T * NH, nullptr);
            ck(cudaDeviceSynchronize(), "prompt");
            std::vector<float> got((size_t) T * NH * HD);
            ck(cudaMemcpy(got.data(), d_op, got.size() * 4, cudaMemcpyDeviceToHost), "prompt out");
            Err ep;
            std::vector<double> want((size_t) NH * HD);
            for (const int r : {0, 1, 55, 63, 64, 65, 127, 128, 255, T - 2, T - 1}) {
                if (r >= T) continue;
                ref_attn(&qp[(size_t) r * NH * HD], kdq, vtrue, pp + r + 1, want.data());
                for (int j = 0; j < NH * HD; ++j) ep.add(got[(size_t) r * NH * HD + j], want[(size_t) j]);
            }
            std::printf("  K8V4 prompt %s: max abs %.3g rel %.3g vs double on the dequantized cells\n",
                        cfg ? "split over 3 key ranges" : "300 rows", ep.abs, ep.rel);
            // the kernel re-quantizes each 32-dim block's probabilities to u8 and sums the stored codes exactly: the
            // error is the format's 4-bit V plus that per-row rounding, the int8 prompt kernel's tolerance class
            report(ep.abs < 3e-2, "gqa_prompt_attn over a K8V4 cache is within its rounding of the fp64 reference");
        }
        cudaFree(d_qp);
        cudaFree(d_op);
        cudaFree(d_split);
    }

    // ---- [5] BF16 queries and output rows: the same launch on the same (bf16-exact) queries, its rows rounded to bf16
    // once.  Both launches round the queries to int8 the same way, so nothing data-dependent stands between them.
    {
        const int T = 300;
        const int64_t pp = cell0 + L - T;
        std::vector<float> qp((size_t) T * NH * HD);
        for (auto& x : qp) x = 2.0f * nd(rng);
        std::vector<uint16_t> qb(qp.size());   // the BF16 form: the query rows rounded to bf16 once, as the chunk path has them
        std::vector<float> qf(qp.size());     // and the exact values back, as the FP32 launch must see them
        for (size_t i = 0; i < qp.size(); ++i) {
            qb[i] = k::bf16_from_f32(qp[i]);
            qf[i] = k::f32_from_bf16(qb[i]);
        }
        float* d_qf = dev<float>(qf.size());
        uint16_t *d_qb = dev<uint16_t>(qb.size()), *d_ob = dev<uint16_t>(qb.size()), *d_rb = dev<uint16_t>(qb.size());
        float* d_of = dev<float>(qf.size());
        ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "qf");
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice), "qb");
        const size_t sf = k::gqa_prompt_split_floats(T, NH, 3);
        float* d_split = dev<float>(sf);
        k::gqa_prompt_attn(d_qf, nullptr, nullptr, NH, NKV, pp, T, d_of, nullptr, k::KvInt8{}, d_split, sf, k::KvFar{}, cache);
        k::to_bf16(d_of, d_rb, (int64_t) T * NH * HD, nullptr);   // the one rounding the BF16 row takes
        k::gqa_prompt_attn_bf16(d_qb, NH, NKV, pp, T, d_ob, nullptr, k::KvInt8{}, d_split, sf, k::KvFar{}, cache);
        ck(cudaDeviceSynchronize(), "prompt bf16");
        std::vector<uint16_t> a((size_t) T * NH * HD), b((size_t) T * NH * HD);
        ck(cudaMemcpy(a.data(), d_rb, a.size() * 2, cudaMemcpyDeviceToHost), "fp32 rows rounded");
        ck(cudaMemcpy(b.data(), d_ob, b.size() * 2, cudaMemcpyDeviceToHost), "bf16 rows");
        size_t differ = 0;
        uint16_t worst = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] == b[i]) continue;
            ++differ;
            const uint16_t d = (uint16_t) std::abs((int) a[i] - (int) b[i]);
            worst = std::max(worst, d);
        }
        std::printf("  K8V4 prompt BF16 rows vs the FP32 launch's rounded: %zu of %zu rows differ (worst %u in bf16 bits)\n",
                    differ, a.size(), (unsigned) worst);
        report(differ == 0, "the BF16 form of the K8V4 prompt kernel is the FP32 launch rounded once, bitwise");
        cudaFree(d_qf); cudaFree(d_qb); cudaFree(d_ob); cudaFree(d_rb); cudaFree(d_of); cudaFree(d_split);
    }

    cudaFree(d_k); cudaFree(d_ks); cudaFree(d_vq); cudaFree(d_ik); cudaFree(d_iv); cudaFree(d_iks); cudaFree(d_ivs);
    cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_qo);
    cudaFree(d_q); cudaFree(d_out); cudaFree(d_scr);
    std::printf("gqa_k8v4_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
