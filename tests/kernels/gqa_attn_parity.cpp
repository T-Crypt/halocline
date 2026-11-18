// tests/kernels/gqa_attn_parity.cpp - gqa_decode_attn and gqa_prompt_attn (halo/kernels/gqa_attn.hpp) and
// attn_qk_prepare (dense_ops.hpp) against a CPU reference in double precision, at the Qwen3.5-27B attention geometry
// (24 query heads, 4 KV heads, 256 wide, 64 rotated dims, interleaved M-RoPE sections 11/11/10), unscaled and under YaRN.
//
// Checks: each kernel's max absolute and relative error to the reference (the int8 prompt kernel's narrow, split and
// 8-warp launches); a decode row's output is bitwise the same
// whatever the window width or the row's place in it; the window kernel (one block a split and KV head) is the 16-row-tile
// launch bitwise at 1 to 9 rows, on every cache form, on a ring and over a batch window's segments; the prompt kernel's
// rows agree with decode's within fp16 rounding.
// Exit 77 without a GPU.  Prints one line per check and a microbench line for the decode and prompt kernels.
#include "halo/kernels/dense_ops.hpp"
#include "halo/kernels/f16_bits.hpp"
#include "halo/kernels/gqa_attn.hpp"

#include "gqa_decode_ref.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;
constexpr double kPi = 3.14159265358979323846;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return (T*) p;
}

float h2f(uint16_t h) { return halo::kernels::f32_from_f16(h); }

// out[h][d] for one query row at `visible` cells
void ref_attn(const float* q, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc, int64_t visible, double* out) {
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        std::vector<double> s((size_t) visible);
        double mx = -1e300;
        for (int64_t c = 0; c < visible; ++c) {
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[h * HD + i] * h2f(kc[((size_t) c * NKV + kvh) * HD + i]);
            s[(size_t) c] = d / 16.0;
            mx = std::max(mx, s[(size_t) c]);
        }
        double l = 0;
        for (auto& x : s) { x = std::exp(x - mx); l += x; }
        for (int i = 0; i < HD; ++i) {
            double a = 0;
            for (int64_t c = 0; c < visible; ++c) a += s[(size_t) c] * h2f(vc[((size_t) c * NKV + kvh) * HD + i]);
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

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;

    // ---- attn_qk_prepare: q/k norm, partial rope with M-RoPE axes, cache write; unscaled, then under YaRN x2 past the
    // trained 262,144 against transformers' YaRN (_compute_yarn_parameters: the frequency blend by pair index, cos and sin
    // times 0.1 ln(factor) + 1), the angle rounded as the kernel takes it (fp32 position times fp32 frequency)
    for (const bool yarn : {false, true}) {
        const int T = 3;
        std::vector<float> qf((size_t) T * NH * 2 * HD), k((size_t) T * NKV * HD), v(k.size()), qn(HD), kn(HD);
        for (auto& x : qf) x = nd(rng);
        for (auto& x : k) x = nd(rng);
        for (auto& x : v) x = nd(rng);
        for (auto& x : qn) x = 1.0f + 0.1f * nd(rng);
        for (auto& x : kn) x = 1.0f + 0.1f * nd(rng);
        std::vector<int32_t> pos = {5, 5, 5, 6, 9, 11, 7, 7, 7};   // a text row, an image row (t, h, w), a text row
        if (yarn) pos = {300001, 300001, 300001, 400000, 400003, 400005, 524287, 524287, 524287};
        halo::kernels::RopeSpec rope;
        rope.rot = ROT;
        rope.theta = 1e7f;
        for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
        const double factor = 2.0, orig = 262144.0, base = 1e7;
        std::vector<double> inv_freq(ROT / 2);
        double mscale = 1.0;
        for (int i = 0; i < ROT / 2; ++i) inv_freq[(size_t) i] = std::pow(base, -2.0 * i / ROT);
        if (yarn) {
            rope.scaling.type = halo::kernels::RopeScalingType::YaRN;
            rope.scaling.factor = factor;
            rope.scaling.orig_ctx = orig;
            rope.scaling.freq_base = base;
            rope.scaling.ext_factor = 1.0;
            // transformers: find_correction_range(beta_fast 32, beta_slow 1, dim, base, orig) with truncation, a linear
            // ramp over dim / 2 pairs, inv = interp (1 - extrap_factor) + extrap extrap_factor, extrap_factor = 1 - ramp
            const auto corr = [&](double rot) { return ROT * std::log(orig / (rot * 2 * kPi)) / (2 * std::log(base)); };
            const double lo = std::max(0.0, std::floor(corr(32))), hi0 = std::min((double) ROT - 1, std::ceil(corr(1)));
            const double hi = lo == hi0 ? hi0 + 0.001 : hi0;
            for (int i = 0; i < ROT / 2; ++i) {
                const double ramp = std::min(1.0, std::max(0.0, (i - lo) / (hi - lo)));
                const double ex = 1.0 - ramp, extrap = inv_freq[(size_t) i];
                inv_freq[(size_t) i] = extrap / factor * (1.0 - ex) + extrap * ex;
            }
            mscale = 0.1 * std::log(factor) + 1.0;
        }
        float *d_qf = dev<float>(qf.size()), *d_k = dev<float>(k.size()), *d_v = dev<float>(v.size()), *d_qn = dev<float>(HD),
              *d_kn = dev<float>(HD), *d_qo = dev<float>((size_t) T * NH * HD);
        int32_t* d_pos = dev<int32_t>(pos.size());
        const int64_t cell0 = 10;
        uint16_t *d_kc = dev<uint16_t>((size_t) (cell0 + T) * NKV * HD), *d_vc = dev<uint16_t>((size_t) (cell0 + T) * NKV * HD);
        ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_qn, qn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_kn, kn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "copy");
        halo::kernels::attn_qk_prepare(d_qf, d_k, d_v, d_qn, d_kn, 1e-6f, rope, d_pos, T, NH, NKV, d_qo, d_kc, d_vc, cell0, nullptr);
        std::vector<float> qo((size_t) T * NH * HD);
        std::vector<uint16_t> kc((size_t) (cell0 + T) * NKV * HD);
        ck(cudaMemcpy(qo.data(), d_qo, qo.size() * 4, cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(kc.data(), d_kc, kc.size() * 2, cudaMemcpyDeviceToHost), "copy");
        Err eq, ek;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NH + NKV; ++j) {
                const bool isq = j < NH;
                const float* x = isq ? &qf[((size_t) t * NH + j) * 2 * HD] : &k[((size_t) t * NKV + (j - NH)) * HD];
                const float* w = isq ? qn.data() : kn.data();
                double ss = 0;
                for (int i = 0; i < HD; ++i) ss += (double) x[i] * x[i];
                const double inv = 1.0 / std::sqrt(ss / HD + 1e-6);
                std::vector<double> y(HD);
                for (int i = 0; i < HD; ++i) y[(size_t) i] = x[i] * inv * w[i];
                std::vector<double> r = y;
                for (int i = 0; i < ROT / 2; ++i) {
                    const int32_t p = pos[(size_t) t * 3 + rope.axis[i]];
                    const double f = yarn ? (double) ((float) p * (float) inv_freq[(size_t) i]) : inv_freq[(size_t) i] * p;
                    const double c = std::cos(f) * mscale, sn = std::sin(f) * mscale;
                    r[(size_t) i] = y[(size_t) i] * c - y[(size_t) i + ROT / 2] * sn;
                    r[(size_t) i + ROT / 2] = y[(size_t) i + ROT / 2] * c + y[(size_t) i] * sn;
                }
                for (int i = 0; i < HD; ++i) {
                    if (isq) eq.add(qo[((size_t) t * NH + j) * HD + i], r[(size_t) i]);
                    else ek.add(h2f(kc[(((size_t) cell0 + t) * NKV + (j - NH)) * HD + i]), r[(size_t) i]);
                }
            }
        std::printf("attn_qk_prepare%s: q max abs %.3g rel %.3g; k (fp16 cache) max abs %.3g rel %.3g\n", yarn ? " (YaRN x2)" : "",
                    eq.abs, eq.rel, ek.abs, ek.rel);
        if (eq.abs > 1e-3 || ek.rel > 2e-3) ++fails;
    }

    // ---- attention: a cache of L cells, decode windows and a prompt chunk
    const int64_t L = 3000, max_cells = 4096;
    std::vector<uint16_t> kc((size_t) max_cells * NKV * HD), vc(kc.size());
    for (auto& x : kc) x = halo::kernels::f16_from_f32(nd(rng));
    for (auto& x : vc) x = halo::kernels::f16_from_f32(nd(rng));
    const int TP = 300;   // the prompt chunk: rows at cells L - TP .. L - 1
    std::vector<float> q((size_t) TP * NH * HD);
    for (auto& x : q) x = 2.0f * nd(rng);
    uint16_t *d_kc = dev<uint16_t>(kc.size()), *d_vc = dev<uint16_t>(vc.size());
    float *d_q = dev<float>(q.size()), *d_out = dev<float>(q.size());
    float* d_scr = dev<float>(halo::kernels::gqa_decode_scratch_floats(max_cells, halo::kernels::kGqaMaxWindow, NH));
    ck(cudaMemcpy(d_kc, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_vc, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy");
    const int64_t p0 = L - TP;
    // reference for the last 8 rows and a few early ones
    const std::vector<int> rows = {0, 1, 57, TP - 8, TP - 7, TP - 6, TP - 5, TP - 4, TP - 3, TP - 2, TP - 1};
    std::vector<std::vector<double>> want(rows.size(), std::vector<double>((size_t) NH * HD));
    for (size_t i = 0; i < rows.size(); ++i) ref_attn(&q[(size_t) rows[i] * NH * HD], kc, vc, p0 + rows[i] + 1, want[i].data());
    // decode: the last 8 rows as windows of 1..8 rows; each row bitwise equal across widths
    constexpr int TW = halo::kernels::kGqaMaxWindow;
    std::vector<float> single((size_t) TW * NH * HD), got((size_t) TW * NH * HD);
    Err ed;
    bool bitwise = true;
    for (int T = 1; T <= TW; ++T) {
        const int r0 = TP - T;
        halo::kernels::gqa_decode_attn(d_q + (size_t) r0 * NH * HD, d_kc, d_vc, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out, nullptr);
        ck(cudaMemcpy(got.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
        // row TP-1 is the window's last row: compare with the single-row window
        const float* last = &got[(size_t) (T - 1) * NH * HD];
        if (T == 1) std::memcpy(single.data(), last, (size_t) NH * HD * 4);
        else if (std::memcmp(single.data(), last, (size_t) NH * HD * 4) != 0) bitwise = false;
        for (int t = 0; t < T; ++t) {
            const int row = r0 + t;
            for (size_t i = 0; i < rows.size(); ++i)
                if (rows[i] == row)
                    for (int j = 0; j < NH * HD; ++j) ed.add(got[(size_t) t * NH * HD + j], want[i][(size_t) j]);
        }
    }
    std::printf("gqa_decode_attn: max abs %.3g rel %.3g vs double; the last row bitwise equal across widths 1-%d: %s\n", ed.abs,
                ed.rel, TW, bitwise ? "yes" : "NO");
    if (ed.abs > 1e-3 || !bitwise) ++fails;   // fp16 q, K, V and p on the tensor cores
    // the window kernel (one block a split and KV head holding every row of the window) is the 16-row-tile launch's
    // rows, bit for bit, at every width from one row to one past its three m16 groups
    {
        bool same = true;
        std::vector<float> wide((size_t) TW * NH * HD), tiled(wide.size());
        for (int T = 1; T <= 9; ++T) {
            const int r0 = TP - T;
            const float* qw = d_q + (size_t) r0 * NH * HD;
            halo::kernels::test::gqa_decode_attn_window(qw, d_kc, d_vc, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out, nullptr);
            ck(cudaMemcpy(wide.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            halo::kernels::test::gqa_decode_attn_tiled(qw, d_kc, d_vc, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out, nullptr);
            ck(cudaMemcpy(tiled.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            if (std::memcmp(wide.data(), tiled.data(), (size_t) T * NH * HD * 4) != 0) same = false;
        }
        std::printf("gqa_decode_attn: the window kernel at 1-9 rows bitwise the 16-row-tile launch: %s\n", same ? "yes" : "NO");
        if (!same) ++fails;
    }
    {
        // a captured window reads its first cell on the device and loops its blocks over the chunks in use: bitwise the
        // host-position launch, past the grid's first round of chunks
        const int64_t L2 = 12000, cap2 = 65536, at = L2 - 8;
        std::vector<uint16_t> k2((size_t) L2 * NKV * HD), v2(k2.size());
        for (auto& x : k2) x = halo::kernels::f16_from_f32(nd(rng));
        for (auto& x : v2) x = halo::kernels::f16_from_f32(nd(rng));
        uint16_t *d_k2 = dev<uint16_t>((size_t) cap2 * NKV * HD), *d_v2 = dev<uint16_t>((size_t) cap2 * NKV * HD);
        float* d_s2 = dev<float>(halo::kernels::gqa_decode_scratch_floats(cap2, 16, NH));
        int32_t* d_at = dev<int32_t>(2);
        const int32_t at2[2] = {(int32_t) at, 0};
        ck(cudaMemcpy(d_k2, k2.data(), k2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v2, v2.data(), v2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_at, at2, sizeof at2, cudaMemcpyHostToDevice), "copy");
        std::vector<float> h1((size_t) 8 * NH * HD), h2(h1.size());
        halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr);
        ck(cudaMemcpy(h1.data(), d_out, h1.size() * 4, cudaMemcpyDeviceToHost), "copy");
        halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr, d_at);
        ck(cudaMemcpy(h2.data(), d_out, h2.size() * 4, cudaMemcpyDeviceToHost), "copy");
        const bool same = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
        std::printf("gqa_decode_attn: device first cell at %lld of %lld cells bitwise the host launch: %s\n", (long long) at,
                    (long long) cap2, same ? "yes" : "NO");
        if (!same) ++fails;
        {   // two sequences' windows in one launch (a batch window's segments), each bitwise its own launch
            const int TA = 5, TB = 7;
            std::vector<float> solo((size_t) (TA + TB) * NH * HD), seg(solo.size());
            halo::kernels::test::gqa_decode_attn_window(d_q, d_k2, d_v2, NH, NKV, 3000, TA, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(solo.data(), d_out, (size_t) TA * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            halo::kernels::test::gqa_decode_attn_window(d_q + (size_t) TA * NH * HD, d_k2, d_v2, NH, NKV, at - TB, TB, d_s2, cap2,
                                                        d_out, nullptr);
            ck(cudaMemcpy(solo.data() + (size_t) TA * NH * HD, d_out, (size_t) TB * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            const int32_t atb[2] = {(int32_t) (at - TB), 0};
            ck(cudaMemcpy(d_at, atb, sizeof atb, cudaMemcpyHostToDevice), "copy");
            halo::kernels::GqaSeg sg[2];
            sg[0].T = TA; sg[0].k_cache = d_k2; sg[0].v_cache = d_v2; sg[0].pos0 = 3000;
            sg[1].T = TB; sg[1].k_cache = d_k2; sg[1].v_cache = d_v2; sg[1].pos0 = at - TB; sg[1].pos_dev = d_at;
            halo::kernels::test::gqa_decode_attn_segs_window(d_q, NH, NKV, sg, 2, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(seg.data(), d_out, seg.size() * 4, cudaMemcpyDeviceToHost), "copy");
            const bool same_seg = std::memcmp(solo.data(), seg.data(), seg.size() * 4) == 0;
            std::printf("gqa_decode_attn_segs: the window kernel's windows of %d and %d rows in one launch bitwise their own "
                        "launches: %s\n", TA, TB, same_seg ? "yes" : "NO");
            if (!same_seg) ++fails;
            // the same launch on the 16-row-tile kernel: a segment with fewer groups than the launch's R is its own rows
            std::vector<float> segt(seg.size());
            halo::kernels::test::gqa_decode_attn_segs_tiled(d_q, NH, NKV, sg, 2, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(segt.data(), d_out, segt.size() * 4, cudaMemcpyDeviceToHost), "copy");
            const bool same_tiled = std::memcmp(seg.data(), segt.data(), seg.size() * 4) == 0;
            std::printf("gqa_decode_attn_segs: the same two windows bitwise the 16-row-tile launch: %s\n", same_tiled ? "yes" : "NO");
            if (!same_tiled) ++fails;
            ck(cudaMemcpy(d_at, at2, sizeof at2, cudaMemcpyHostToDevice), "copy");
        }
        // its cost at a short context in a large capacity, against the host launch's grid of the chunks in use
        const int32_t near_at[2] = {2999, 0};
        ck(cudaMemcpy(d_at, near_at, sizeof near_at, cudaMemcpyHostToDevice), "copy");
        double us[2];
        for (int dv = 0; dv < 2; ++dv) {
            for (int i = 0; i < 5; ++i)
                halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i)
                halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            us[dv] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
        }
        std::printf("bench: decode 1 row x 3000 cells in %lld: %.1f us host first cell, %.1f us device first cell\n",
                    (long long) cap2, us[0], us[1]);
        for (int T : {1, 8, 16}) {   // a window of T rows near 12K cells
            for (int i = 0; i < 5; ++i) halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at - 8, T, d_s2, cap2, d_out, nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 100; ++i) halo::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at - 8, T, d_s2, cap2, d_out, nullptr);
            cudaDeviceSynchronize();
            std::printf("bench: decode %d rows x %lld cells: %.1f us\n", T, (long long) at, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 100);
        }
        cudaFree(d_k2); cudaFree(d_v2); cudaFree(d_s2); cudaFree(d_at);
    }
    {
        // A ring (the windowed drafter's cache, cell p at p % cap) laps in: the same window read from the ring and from an
        // absolute cache agree, and every output is finite. While the ring's sweep started at chunk 0, a window past
        // position cap read no cell at all (sum 0, NaN out): the drafter failure at depth 140,288. The window's first
        // cell lands at every offset of its chunk across the positions tried (the extra chunk's case).
        constexpr int64_t W = 1000, CAP = W + halo::kernels::kGqaMaxWindow - 1, L3 = 5 * CAP;
        std::vector<uint16_t> ka((size_t) L3 * NKV * HD), va(ka.size());
        for (auto& x : ka) x = halo::kernels::f16_from_f32(nd(rng));
        for (auto& x : va) x = halo::kernels::f16_from_f32(nd(rng));
        uint16_t *d_ka = dev<uint16_t>(ka.size()), *d_va = dev<uint16_t>(va.size());
        uint16_t *d_kr = dev<uint16_t>((size_t) CAP * NKV * HD), *d_vr = dev<uint16_t>((size_t) CAP * NKV * HD);
        float* d_s3 = dev<float>(halo::kernels::gqa_decode_scratch_floats(L3, halo::kernels::kGqaMaxWindow, NH));
        ck(cudaMemcpy(d_ka, ka.data(), ka.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_va, va.data(), va.size() * 2, cudaMemcpyHostToDevice), "copy");
        const size_t cell = (size_t) NKV * HD;
        std::vector<uint16_t> kr((size_t) CAP * cell), vr(kr.size());
        std::vector<float> ga((size_t) halo::kernels::kGqaMaxWindow * NH * HD), gr(ga.size()), gt(ga.size());
        Err er;
        bool finite = true, tiled_same = true;
        int tried = 0;
        for (int T : {1, 7, 16}) {
            for (int64_t pos0 = 3 * CAP + 5; pos0 < 3 * CAP + 5 + 2 * halo::kernels::kGqaCellChunk; pos0 += 3) {
                // the ring as a session leaves it: every position up to the window's last, written in order
                for (int64_t p = 0; p < pos0 + T; ++p) {
                    std::memcpy(&kr[(size_t) (p % CAP) * cell], &ka[(size_t) p * cell], cell * 2);
                    std::memcpy(&vr[(size_t) (p % CAP) * cell], &va[(size_t) p * cell], cell * 2);
                }
                ck(cudaMemcpy(d_kr, kr.data(), kr.size() * 2, cudaMemcpyHostToDevice), "copy");
                ck(cudaMemcpy(d_vr, vr.data(), vr.size() * 2, cudaMemcpyHostToDevice), "copy");
                halo::kernels::test::gqa_decode_attn_window(d_q, d_ka, d_va, NH, NKV, pos0, T, d_s3, L3, d_out, nullptr, nullptr, {},
                                                            nullptr, nullptr, {}, {}, W, 0);
                ck(cudaMemcpy(ga.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                halo::kernels::test::gqa_decode_attn_window(d_q, d_kr, d_vr, NH, NKV, pos0, T, d_s3, L3, d_out, nullptr, nullptr, {},
                                                            nullptr, nullptr, {}, {}, W, CAP);
                ck(cudaMemcpy(gr.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                halo::kernels::test::gqa_decode_attn_tiled(d_q, d_kr, d_vr, NH, NKV, pos0, T, d_s3, L3, d_out, nullptr, nullptr, {},
                                                           nullptr, nullptr, {}, {}, W, CAP);
                ck(cudaMemcpy(gt.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                if (std::memcmp(gr.data(), gt.data(), (size_t) T * NH * HD * 4) != 0) tiled_same = false;
                for (size_t j = 0; j < (size_t) T * NH * HD; ++j) {
                    if (!std::isfinite(gr[j])) finite = false;
                    er.add(gr[j], ga[j]);
                }
                ++tried;
            }
        }
        std::printf("gqa_decode_attn: a ring of %lld cells near position %lld (%d windows of 1/7/16 rows, the window kernel) vs "
                    "the absolute cache: max abs %.3g, all finite: %s, bitwise the 16-row-tile launch: %s\n",
                    (long long) CAP, (long long) (3 * CAP), tried, er.abs, finite ? "yes" : "NO", tiled_same ? "yes" : "NO");
        if (!finite || er.abs > 1e-3 || !tiled_same) ++fails;
        cudaFree(d_ka); cudaFree(d_va); cudaFree(d_kr); cudaFree(d_vr); cudaFree(d_s3);
    }
    // prompt chunk
    halo::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    std::vector<float> pr(q.size());
    ck(cudaMemcpy(pr.data(), d_out, pr.size() * 4, cudaMemcpyDeviceToHost), "copy");
    Err ep;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < NH * HD; ++j) ep.add(pr[(size_t) rows[i] * NH * HD + j], want[i][(size_t) j]);
    std::printf("gqa_prompt_attn: max abs %.3g rel %.3g vs double (fp16 q and p)\n", ep.abs, ep.rel);
    if (ep.abs > 5e-3) ++fails;

    // ---- int8 cache: the same cells as codes and per-64 scales (the write kernel's rule), against the reference on the
    // dequantized values
    {
        const size_t rows = (size_t) max_cells * NKV;
        std::vector<int8_t> k8(rows * HD), v8(rows * HD);
        std::vector<uint16_t> ks(rows * HD / 64), vs(rows * HD / 64), kd(kc.size()), vd(vc.size());
        auto quant = [&](const std::vector<uint16_t>& src, std::vector<int8_t>& q, std::vector<uint16_t>& sc, std::vector<uint16_t>& deq) {
            for (size_t g = 0; g < rows * HD / 64; ++g) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(h2f(src[g * 64 + i])));
                const uint16_t h = halo::kernels::f16_from_f32(amax / 127.0f);
                const float s16 = h2f(h);
                sc[g] = h;
                for (int i = 0; i < 64; ++i) {
                    const float c = s16 > 0 ? std::rint(h2f(src[g * 64 + i]) / s16) : 0.0f;
                    q[g * 64 + i] = (int8_t) std::max(-127.0f, std::min(127.0f, c));
                    deq[g * 64 + i] = halo::kernels::f16_from_f32(q[g * 64 + i] * s16);
                }
            }
        };
        quant(kc, k8, ks, kd);
        quant(vc, v8, vs, vd);
        halo::kernels::KvInt8 q8;
        ck(cudaMalloc(&q8.k, k8.size()), "malloc");
        ck(cudaMalloc(&q8.v, v8.size()), "malloc");
        ck(cudaMalloc(&q8.ks, ks.size() * 2), "malloc");
        ck(cudaMalloc(&q8.vs, vs.size() * 2), "malloc");
        cudaMemcpy(q8.k, k8.data(), k8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.v, v8.data(), v8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice);
        // the prompt kernel rounds the queries to int8 per 64 dims (amax / 127) for its int8 QK: references on the
        // queries as given and on those
        auto ref8 = [&](const float* qrow, int64_t visible, double* w, double* wq) {
            ref_attn(qrow, kd, vd, visible, w);
            std::vector<float> qq(qrow, qrow + (size_t) NH * HD);
            for (size_t g0 = 0; g0 < qq.size(); g0 += 64) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(qq[g0 + i]));
                for (int i = 0; i < 64; ++i)
                    qq[g0 + i] = amax > 0 ? (float) std::nearbyint(qq[g0 + i] * (127.0f / amax)) * (amax / 127.0f) : 0.0f;
            }
            ref_attn(qq.data(), kd, vd, visible, wq);
        };
        std::vector<double> w8((size_t) NH * HD), w8q((size_t) NH * HD);
        ref8(&q[(size_t) (TP - 1) * NH * HD], p0 + TP, w8.data(), w8q.data());
        halo::kernels::gqa_decode_attn(d_q + (size_t) (TP - 1) * NH * HD, nullptr, nullptr, NH, NKV, p0 + TP - 1, 1, d_scr,
                                         max_cells, d_out, nullptr, nullptr, q8);
        std::vector<float> g8((size_t) NH * HD);
        ck(cudaMemcpy(g8.data(), d_out, g8.size() * 4, cudaMemcpyDeviceToHost), "copy");
        Err e8;
        for (int j = 0; j < NH * HD; ++j) e8.add(g8[(size_t) j], w8[(size_t) j]);
        std::printf("int8 cache: decode max abs %.3g rel %.3g vs double on the dequantized cells\n", e8.abs, e8.rel);
        // decode reads code x scale in fp32, the reference its fp16 rounding
        if (e8.abs > 1e-3) ++fails;
        {   // the same widths on the int8 cache: the window kernel is the 16-row-tile launch's rows, bit for bit
            bool same = true;
            std::vector<float> wide((size_t) halo::kernels::kGqaMaxWindow * NH * HD), tiled(wide.size());
            for (int T = 1; T <= 9; ++T) {
                const int r0 = TP - T;
                const float* qw = d_q + (size_t) r0 * NH * HD;
                halo::kernels::test::gqa_decode_attn_window(qw, nullptr, nullptr, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out,
                                                            nullptr, nullptr, q8);
                ck(cudaMemcpy(wide.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                halo::kernels::test::gqa_decode_attn_tiled(qw, nullptr, nullptr, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out, nullptr,
                                                           nullptr, q8);
                ck(cudaMemcpy(tiled.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                if (std::memcmp(wide.data(), tiled.data(), (size_t) T * NH * HD * 4) != 0) same = false;
            }
            std::printf("int8 cache: the window kernel at 1-9 rows bitwise the 16-row-tile launch: %s\n", same ? "yes" : "NO");
            if (!same) ++fails;
        }

        // prompt chunks at the end of the L cells: 300 rows (narrow: 4-warp blocks), the same with split scratch (the
        // cells split over more blocks and merged), and 2048 rows (8-warp blocks), each at rows on and around the block
        // and key-tile edges
        const int TB = 2048;
        std::vector<float> qb((size_t) TB * NH * HD);
        for (auto& x : qb) x = 2.0f * nd(rng);
        float *d_qb = dev<float>(qb.size()), *d_ob = dev<float>(qb.size());
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 4, cudaMemcpyHostToDevice), "copy");
        const size_t split_floats = halo::kernels::gqa_prompt_split_floats(TP, NH, 8);
        float* d_split = dev<float>(split_floats);
        for (int cfg = 0; cfg < 3; ++cfg) {
            const int T = cfg < 2 ? TP : TB;
            const int64_t pp = L - T;
            const float* qh = cfg < 2 ? q.data() : qb.data();
            halo::kernels::gqa_prompt_attn(cfg < 2 ? d_q : d_qb, nullptr, nullptr, NH, NKV, pp, T, d_ob, nullptr, q8,
                                             cfg == 1 ? d_split : nullptr, cfg == 1 ? split_floats : 0);
            std::vector<float> po((size_t) T * NH * HD);
            ck(cudaMemcpy(po.data(), d_ob, po.size() * 4, cudaMemcpyDeviceToHost), "copy");
            Err p8, p8q;
            for (const int r : {0, 1, 57, 63, 64, 65, 127, 128, 255, T - 2, T - 1, 1000, 2047}) {
                if (r >= T) continue;
                ref8(&qh[(size_t) r * NH * HD], pp + r + 1, w8.data(), w8q.data());
                for (int j = 0; j < NH * HD; ++j) {
                    p8.add(po[(size_t) r * NH * HD + j], w8[(size_t) j]);
                    p8q.add(po[(size_t) r * NH * HD + j], w8q[(size_t) j]);
                }
            }
            static const char* what[3] = {"300 rows", "300 rows split", "2048 rows"};
            std::printf("int8 cache: prompt %s max abs %.3g (%.3g against int8-rounded queries) vs double on the dequantized "
                        "cells\n", what[cfg], p8.abs, p8q.abs);
            // the prompt's QK reads int8 queries; its P V rounds each probability times its cell's scale to u8 per row per
            // 64-cell tile (half a step of 1/255 of the tile's largest product each), summed exactly in int32
            if (p8q.abs > 2e-2 || p8.abs > 3e-2) ++fails;
        }
        cudaFree(d_qb);
        cudaFree(d_ob);
        cudaFree(d_split);
    }

    // ---- microbench: one decode row at 3000 cells, and the 300-row chunk
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int reps = 50;
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i)
        halo::kernels::gqa_decode_attn(d_q, d_kc, d_vc, NH, NKV, L - 1, 1, d_scr, max_cells, d_out, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double us_dec = 1000.0 * ms / reps, bytes = (double) L * NKV * HD * 2 * 2;
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) halo::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    const double us_pr = 1000.0 * ms / 10;
    double flops = 0;
    for (int t = 0; t < TP; ++t) flops += 4.0 * NH * HD * (double) (p0 + t + 1);
    std::printf("bench: decode 1 row x %lld cells %.1f us (%.0f GB/s of K/V); prompt %d rows %.1f us (%.1f TFLOP/s)\n",
                (long long) L, us_dec, bytes / us_dec / 1e3, TP, us_pr, flops / us_pr / 1e6);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {   // a 2048-row chunk at the end of 4K and 16K caches, fp16 and int8
        for (const int64_t cells : {4096, 16384}) {
            const int64_t T2 = 2048, pp = cells - T2;
            uint16_t* bk = dev<uint16_t>((size_t) cells * NKV * HD);
            uint16_t* bv = dev<uint16_t>((size_t) cells * NKV * HD);
            float* bq = dev<float>((size_t) T2 * NH * HD);
            float* bo = dev<float>((size_t) T2 * NH * HD);
            halo::kernels::KvInt8 b8;
            cudaMalloc(&b8.k, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.v, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.ks, (size_t) cells * NKV * HD / 64 * 2);
            cudaMalloc(&b8.vs, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(bk, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bv, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bq, 0x3C, (size_t) T2 * NH * HD * 4);
            cudaMemset(b8.k, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.v, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.ks, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(b8.vs, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            double fl = 0;
            for (int64_t t = 0; t < T2; ++t) fl += 4.0 * NH * HD * (double) (pp + t + 1);
            // fp16, int8, int8 with scratch for three key splits
            const size_t sf = halo::kernels::gqa_prompt_split_floats(T2, NH, 3);
            float* bs = dev<float>(sf);
            for (int run = 0; run < 3; ++run) {
                const halo::kernels::KvInt8 kv = run ? b8 : halo::kernels::KvInt8{};
                float* scr = run == 2 ? bs : nullptr;
                halo::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(a);
                for (int i = 0; i < 5; ++i) halo::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                cudaEventElapsedTime(&ms, a, b);
                std::printf("bench: prompt chunk 2048 rows at %lld cells, %s cache: %.2f ms (%.1f TFLOP/s)\n", (long long) cells,
                            run == 0 ? "fp16" : run == 1 ? "int8" : "int8 split scratch", ms / 5, fl / (ms / 5) / 1e9);
            }
            cudaFree(bs);
            cudaFree(bk); cudaFree(bv); cudaFree(bq); cudaFree(bo); cudaFree(b8.k); cudaFree(b8.v); cudaFree(b8.ks); cudaFree(b8.vs);
        }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
