// tests/kernels/gqa_k8v4_parity.cpp - the dense path's K8V4 cache (halo/kernels/gqa_attn.hpp: K in the int8 form, V in
// rotated q4_0 blocks) against a double-precision CPU reference, at the Qwen3.5-27B attention geometry (24 query heads,
// 4 KV heads, 256 wide, 64 rotated dims, M-RoPE sections 11/11/10).
//
// Checks:
//   1. attn_qk_prepare's K8V4 write: K's codes and per-64 scales bitwise the host rule (the int8 cache's own), V's blocks
//      bitwise the host quantizer over the host-rotated row;
//   2. the V rotation inside the write is fwht256_cuda's bits (the same row through the shared kernel);
//   3. gqa_decode_attn + the output's inverse rotation against the fp64 reference on the dequantized cells, and a row's
//      bits equal whatever the window width;
//   4. gqa_prompt_attn (4-warp, and split over two key ranges) + the inverse rotation, at rows on and around the tile
//      and query-block edges, against the same reference;
//   5. the same cache read as BF16 queries and output rows is one rounding of the fp32 launch's rows.
// Exit 77 without a GPU.  GPU test: build it, do not run it while a production engine holds the card.
#include "halo/kernels/bf16_bits.hpp"
#include "halo/kernels/dense_ops.hpp"
#include "halo/kernels/f16_bits.hpp"
#include "halo/kernels/gqa_attn.hpp"
#include "halo/kernels/kv_q4_layout.hpp"

#include <cuda_runtime.h>

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
    std::vector<float> K((size_t) L * NKV * HD), V(K.size());
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    // ---- the host reference: K by the dense int8 rule, V rotated then q4_0 (halo/kernels/kv_q4_layout.hpp)
    std::vector<int8_t> rk(K.size());
    std::vector<uint16_t> rks((size_t) L * NKV * (HD / 64));
    for (size_t row = 0; row < (size_t) L * NKV; ++row)
        for (int g = 0; g < HD / 64; ++g) {
            float amax = 0.0f;
            for (int e = 0; e < 64; ++e) amax = std::max(amax, std::fabs(K[row * HD + g * 64 + e]));
            const uint16_t sb = k::f16_from_f32(amax / 127.0f);
            const float sf = h2f(sb);
            rks[row * (HD / 64) + g] = sb;
            for (int e = 0; e < 64; ++e) {
                const float x = K[row * HD + g * 64 + e];
                int q = 0;
                if (sf > 0.0f) q = std::max(-127, std::min(127, (int) std::nearbyint(x / sf)));
                rk[row * HD + g * 64 + e] = (int8_t) q;
            }
        }
    std::vector<float> vrot(V);
    for (int64_t row = 0; row < L * NKV; ++row) fwht_host(&vrot[(size_t) row * HD]);
    std::vector<k::KvQ4Block> rv4((size_t) L * NKV * (HD / k::kKvQ4Block));
    for (int64_t row = 0; row < L * NKV; ++row)
        for (int b = 0; b < HD / k::kKvQ4Block; ++b)
            rv4[(size_t) row * (HD / k::kKvQ4Block) + b] = k::kv_q4_quantize_host(&vrot[(size_t) row * HD + b * k::kKvQ4Block]);

    // ---- device cache, written by attn_qk_prepare's K8V4 form
    int8_t* d_k = dev<int8_t>((size_t) max_cells * NKV * HD);
    uint16_t* d_ks = dev<uint16_t>((size_t) max_cells * NKV * (HD / 64));
    uint8_t* d_vq = dev<uint8_t>((size_t) max_cells * NKV * k::kv_q4_head_bytes(HD));
    float *d_kf = dev<float>(K.size()), *d_vf = dev<float>(V.size()), *d_qo = dev<float>((size_t) L * NH * HD);
    k::KvK8V4 cache{d_k, d_ks, d_vq};
    const int64_t cell0 = 37;   // the write lands at 37 .. 37 + L - 1
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
        k::RopeSpec rope;
        rope.rot = ROT;
        rope.theta = 1e7f;
        for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
        k::attn_qk_prepare(d_qf, d_kf, d_vf, d_qn, d_kn, 1e-6f, rope, d_pos, L, NH, NKV, d_qo, nullptr, nullptr, cell0, nullptr, 256,
                           true, 0, nullptr, k::KvInt8{}, k::KvFar{}, cache);
        ck(cudaDeviceSynchronize(), "prepare");
        cudaFree(d_qf); cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_pos);
    }
    // [1] the written bytes, cell by cell
    {
        std::vector<int8_t> hk((size_t) L * NKV * HD);
        std::vector<uint16_t> hks((size_t) L * NKV * (HD / 64));
        std::vector<uint8_t> hv((size_t) L * NKV * k::kv_q4_head_bytes(HD));
        ck(cudaMemcpy(hk.data(), d_k + (size_t) cell0 * NKV * HD, hk.size(), cudaMemcpyDeviceToHost), "k");
        ck(cudaMemcpy(hks.data(), d_ks + (size_t) cell0 * NKV * (HD / 64), hks.size() * 2, cudaMemcpyDeviceToHost), "ks");
        ck(cudaMemcpy(hv.data(), d_vq + (size_t) cell0 * NKV * k::kv_q4_head_bytes(HD), hv.size(), cudaMemcpyDeviceToHost), "v");
        int bad = 0;
        for (size_t i = 0; i < hk.size(); ++i) bad += hk[i] != rk[i];
        for (size_t i = 0; i < hks.size(); ++i) bad += hks[i] != rks[i];
        for (size_t i = 0; i < hv.size(); ++i) bad += hv[i] != reinterpret_cast<const uint8_t*>(rv4.data())[i];
        std::printf("  K8V4 write: K %s, V %s (%d bytes differ of %zu)\n", bad ? "differs" : "bitwise",
                    bad ? "differs" : "bitwise", bad, hk.size() + hks.size() * 2 + hv.size());
        report(bad == 0, "attn_qk_prepare's K8V4 cache write matches the host rule bitwise");
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

    // ---- the dequantized cells for the attention reference: K as stored, V un-rotated back
    std::vector<float> kdq((size_t) L * NKV * HD), vdq((size_t) L * NKV * HD);
    for (int64_t row = 0; row < L * NKV; ++row)
        for (int d = 0; d < HD; ++d) {
            kdq[(size_t) row * HD + d] = (float) rk[(size_t) row * HD + d] * h2f(rks[(size_t) row * (HD / 64) + d / 64]);
            int codes[8];
            k::kv_q4_codes8(rv4[(size_t) row * 8 + d / 32].qs, (d & 31) & ~7, codes);
            vdq[(size_t) row * HD + d] = k::kv_q4_dequant(codes[d & 7], h2f(rv4[(size_t) row * 8 + d / 32].d));
        }
    std::vector<float> vtrue(vdq);
    for (int64_t row = 0; row < L * NKV; ++row) fwht_host(&vtrue[(size_t) row * HD]);

    // ---- [3] decode windows over the K8V4 cache, the output rotated back once per head
    std::vector<float> q((size_t) k::kGqaMaxWindow * NH * HD);
    for (auto& x : q) x = 2.0f * nd(rng);
    float* d_q = dev<float>(q.size());
    float* d_out = dev<float>(q.size());
    float* d_scr = dev<float>(k::gqa_decode_scratch_floats(max_cells, k::kGqaMaxWindow, NH));
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
    const int64_t p0 = cell0 + L - (int64_t) k::kGqaMaxWindow;
    std::vector<double> want((size_t) NH * HD);
    Err ed;
    bool bitwise = true, have = false;
    std::vector<float> first((size_t) NH * HD);
    for (int T = 1; T <= k::kGqaMaxWindow; ++T) {
        k::gqa_decode_attn(d_q, nullptr, nullptr, NH, NKV, p0, T, d_scr, max_cells, d_out, nullptr, nullptr, k::KvInt8{},
                           nullptr, nullptr, k::KvFar{}, cache);
        k::fwht256_inplace_cuda(d_out, (int64_t) T * NH, nullptr);   // the engine's inverse rotation of each head
        ck(cudaDeviceSynchronize(), "decode");
        std::vector<float> got((size_t) T * NH * HD);
        ck(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "decode out");
        const float* last = &got[(size_t) (T - 1) * NH * HD];
        if (!have) { std::memcpy(first.data(), last, first.size() * 4); have = true; }
        else if (std::memcmp(first.data(), last, first.size() * 4) != 0) bitwise = false;
        ref_attn(&q[(size_t) (T - 1) * NH * HD], kdq, vtrue, p0 + T, want.data());
        for (int j = 0; j < NH * HD; ++j) ed.add(last[j], want[(size_t) j]);
    }
    std::printf("  K8V4 decode: max abs %.3g rel %.3g vs double; the last row bitwise equal across widths: %s\n", ed.abs, ed.rel,
                bitwise ? "yes" : "NO");
    report(ed.abs < 2e-3 && bitwise, "gqa_decode_attn over a K8V4 cache matches the fp64 reference");

    // ---- [4] prompt chunks: 300 rows at the end of the cache, plain and split over two key ranges
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

    // ---- [5] BF16 queries and output rows are the fp32 launch rounded once
    {
        const int T = 300;
        const int64_t pp = cell0 + L - T;
        std::vector<float> qp((size_t) T * NH * HD);
        for (auto& x : qp) x = 2.0f * nd(rng);
        std::vector<uint16_t> qb(qp.size());   // the BF16 form: the query rows rounded to bf16 once, as the chunk path has them
        for (size_t i = 0; i < qp.size(); ++i) qb[i] = k::bf16_from_f32(qp[i]);
        float* d_qp = dev<float>(qp.size());
        uint16_t *d_qb = dev<uint16_t>(qb.size()), *d_ob = dev<uint16_t>(qb.size());
        ck(cudaMemcpy(d_qp, qp.data(), qp.size() * 4, cudaMemcpyHostToDevice), "qp");
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice), "qb");
        const size_t sf = k::gqa_prompt_split_floats(T, NH, 3);
        float* d_split = dev<float>(sf);
        float* d_of = dev<float>((size_t) T * NH * HD);
        k::gqa_prompt_attn(d_qp, nullptr, nullptr, NH, NKV, pp, T, d_of, nullptr, k::KvInt8{}, d_split, sf, k::KvFar{}, cache);
        k::fwht256_inplace_cuda(d_of, (int64_t) T * NH, nullptr);
        k::gqa_prompt_attn_bf16(d_qb, NH, NKV, pp, T, d_ob, nullptr, k::KvInt8{}, d_split, sf, k::KvFar{}, cache);
        // the engine rotates the BF16 rows as fp32 (to_f32 then fwht): the same order here
        float* d_tmp = dev<float>((size_t) T * NH * HD);
        k::to_f32(d_ob, d_tmp, (int64_t) T * NH * HD, nullptr);
        k::fwht256_inplace_cuda(d_tmp, (int64_t) T * NH, nullptr);
        ck(cudaDeviceSynchronize(), "prompt bf16");
        std::vector<float> a((size_t) T * NH * HD), b((size_t) T * NH * HD);
        ck(cudaMemcpy(a.data(), d_of, a.size() * 4, cudaMemcpyDeviceToHost), "fp32 out");
        ck(cudaMemcpy(b.data(), d_tmp, b.size() * 4, cudaMemcpyDeviceToHost), "bf16 out");
        double worst = 0;
        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
        std::printf("  K8V4 prompt BF16 rows vs FP32 rows: max abs %.3g (one BF16 query rounding, one BF16 output rounding)\n", worst);
        report(worst < 2e-2, "the BF16 form of the K8V4 prompt kernel is the FP32 launch rounded");
        cudaFree(d_qp); cudaFree(d_qb); cudaFree(d_ob); cudaFree(d_tmp); cudaFree(d_of); cudaFree(d_split);
    }

    cudaFree(d_k); cudaFree(d_ks); cudaFree(d_vq); cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_qo);
    cudaFree(d_q); cudaFree(d_out); cudaFree(d_scr);
    std::printf("gqa_k8v4_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
