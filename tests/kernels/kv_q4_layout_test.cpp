// tests/kernels/kv_q4_layout_test.cpp - the q4_0 KV block's rule on the CPU (halo/kernels/kv_q4_layout.hpp): the scale is
// the block's signed largest-|value| over -8, the codes are ggml's trunc(x / d + 8.5) clamped to 0..15, the stored code's
// value is (code - 8) * d, and the eight-code reader every kernel uses agrees with the element layout. The dense path's
// K8V4 and rotated 4-bit caches and the sparse path's q4_0 cache all quantize through this header, so a change here that
// moves a bit moves both paths' bytes.
//
// No GPU, no driver: a CPU test (`ctest -L cpu`).
#include "halo/kernels/kv_q4_layout.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace k = halo::kernels;

namespace {
int g_fail = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

float f16(uint16_t h) { return k::f32_from_f16(h); }

// The element layout written out: element j is the low nibble of qs[j] for j < 16 and the high nibble of qs[j - 16]
// from 16 on.  The oracle the kernel-facing reader is checked against.
int element_code(const k::KvQ4Block& b, int j) {
    return j < 16 ? (b.qs[j] & 0x0F) : ((b.qs[j - 16] >> 4) & 0x0F);
}
}  // namespace

int main() {
    std::printf("kv_q4_layout: block %zu bytes, %d values, head bytes 256 -> %llu, 128 -> %llu\n", sizeof(k::KvQ4Block),
                k::kKvQ4Block, (unsigned long long) k::kv_q4_head_bytes(256), (unsigned long long) k::kv_q4_head_bytes(128));
    expect(sizeof(k::KvQ4Block) == 18, "a block is 18 bytes (an fp16 scale and 16 code bytes)");
    expect(k::kv_q4_head_bytes(256) == 8 * 18, "a 256-wide head row is eight blocks");
    expect(k::kv_q4_head_bytes(128) == 4 * 18, "a 128-wide head row is four blocks");

    std::mt19937 rng(20261117u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---- the scale: the signed largest-|value| over -8, ties to the larger value; codes against the unrounded delta
    {
        float x[32];
        for (int round = 0; round < 200; ++round) {
            for (float& v : x) v = nd(rng) * 3.0f;
            const k::KvQ4Block b = k::kv_q4_quantize_host(x);
            float amax = 0.0f, mval = 0.0f;
            for (float v : x)
                if (std::fabs(v) > amax || (std::fabs(v) == amax && v > mval)) {
                    amax = std::fabs(v);
                    mval = v;
                }
            expect(b.d == k::kv_q4_scale_bits(mval), "the stored scale is f16(largest-|value| / -8)");
            const float d = k::kv_q4_delta(mval);
            const float ds = f16(b.d);
            for (int j = 0; j < 32; ++j) {
                expect(element_code(b, j) == k::kv_q4_code(x[j], d), "each code is trunc(x / d + 8.5) against the delta");
                // the stored (fp16) scale decides the value: half a step for a value inside the range, a whole step at
                // the clipped end (the largest positive value maps to code 0, and its negative twin to 15)
                // the codes are rounded against the delta, the value is (code - 8) times the STORED fp16 scale, so a
                // value's error is half a step plus the scale's own fp16 rounding (|x| * 2^-11 at worst)
                const float back = k::kv_q4_dequant(element_code(b, j), ds);
                const float slack = std::fabs(x[j]) * (1.0f / 1024.0f) + 1e-6f;
                expect(std::fabs(back - x[j]) <= (std::fabs(ds) + slack), "a value is within one step of its code's value");
                // inside the codes' span (|x / d| <= 7.5) the code is the nearest; past it the top code clips, and the
                // error is up to one step. That asymmetry is ggml's q4_0 (the delta is max / -8), not a bug here.
                if (std::fabs(x[j] / d) <= 7.5f)
                    expect(std::fabs(back - x[j]) <= 0.5f * std::fabs(ds) + slack, "an in-range value is within half a step");
            }
        }
    }

    // ---- ties resolve to the larger value, in any order, so a warp's shuffle reduction and a host loop agree
    {
        float a[32] = {0.0f}, b[32] = {0.0f};
        a[0] = 2.0f, a[5] = -2.0f;
        b[5] = -2.0f, b[31] = 2.0f;
        const k::KvQ4Block qa = k::kv_q4_quantize_host(a), qb = k::kv_q4_quantize_host(b);
        expect(qa.d == qb.d, "a tie on |x| picks one scale whatever the order");
        expect(f16(qa.d) == -0.25f, "the tie picks the larger value (+2 / -8), not the smaller");
        expect(element_code(qa, 0) == 0 && element_code(qa, 5) == 15, "the largest value is code 0 and its twin clips to 15");
        expect(k::kv_q4_dequant(0, f16(qa.d)) == 2.0f, "code 0 is the largest value exactly");
    }

    // ---- an all-zero block: a zero scale, every code 8, every value 0
    {
        float z[32] = {0.0f};
        const k::KvQ4Block b = k::kv_q4_quantize_host(z);
        expect(f16(b.d) == 0.0f, "an all-zero block stores a zero scale");
        for (int j = 0; j < 32; ++j) expect(element_code(b, j) == 8, "an all-zero block stores code 8");
        expect(k::kv_q4_dequant(8, f16(b.d)) == 0.0f, "code 8 at a zero scale is zero");
    }

    // ---- the eight-code reader: every dim of a 256-wide head row, read the way the decode and prompt kernels read it
    {
        float row[256];
        for (float& v : row) v = nd(rng) * 0.5f;
        k::KvQ4Block blocks[8];
        for (int b = 0; b < 8; ++b) blocks[b] = k::kv_q4_quantize_host(&row[b * 32]);
        int bad = 0;
        for (int d = 0; d < 256; ++d) {
            const int d8 = (d & 31) & ~7;      // the reader's granularity: eight dims from a multiple of 8
            int c[8];
            k::kv_q4_codes8(blocks[d >> 5].qs, d8, c);
            if (c[d & 7] != element_code(blocks[d >> 5], d & 31)) ++bad;
            if (k::kv_q4_dequant(c[d & 7], f16(blocks[d >> 5].d)) !=
                k::kv_q4_dequant(element_code(blocks[d >> 5], d & 31), f16(blocks[d >> 5].d)))
                ++bad;
        }
        expect(bad == 0, "kv_q4_codes8 reads every dim the same code and value the element layout does");
        // and a 16-dim piece of a row is the way the prompt kernel's tile copy reads it
        bad = 0;
        for (int c16 = 0; c16 < 16; ++c16) {
            const k::KvQ4Block& blk = blocks[c16 >> 1];
            int c[16];
            k::kv_q4_codes8(blk.qs, (c16 & 1) * 16, c);
            k::kv_q4_codes8(blk.qs, (c16 & 1) * 16 + 8, c + 8);
            for (int e = 0; e < 16; ++e)
                if (c[e] != element_code(blk, (c16 & 1) * 16 + e)) ++bad;
        }
        expect(bad == 0, "a 16-dim piece of a row reads the codes of its own dims");
    }

    // ---- the rotation the dense 4-bit caches apply: the 256-point transform with the 1/16 at the load (the stages and
    // order of fwht256_cuda, and of the in-block copy at the cache write) is orthonormal and its own inverse. Rotating
    // a query and a key therefore leaves their dot product alone, and rotating a V row is undone by the same call on
    // the attention output. The dense kernels rotate K after its rope and V at the cache write, and Q before attention.
    {
        const auto fwht = [](float* x) {
            for (int s = 1; s < 256; s *= 2)
                for (int i = 0; i < 256; i += 2 * s)
                    for (int j = 0; j < s; ++j) {
                        const float u = x[i + j], w = x[i + j + s];
                        x[i + j] = u + w;
                        x[i + j + s] = u - w;
                    }
            for (int i = 0; i < 256; ++i) x[i] *= 1.0f / 16.0f;
        };
        std::mt19937 r(20261117u);
        std::normal_distribution<float> n(0.0f, 1.0f);
        double worst_norm = 0.0, worst_inv = 0.0, worst_dot = 0.0, worst_rt = 0.0;
        for (int round = 0; round < 100; ++round) {
            float a[256], b[256], a0[256], b0[256];
            for (int i = 0; i < 256; ++i) a[i] = a0[i] = n(r), b[i] = b0[i] = n(r);
            fwht(a);
            fwht(b);
            double dot = 0.0, dot0 = 0.0, na = 0.0, nb = 0.0, na0 = 0.0;
            for (int i = 0; i < 256; ++i) {
                dot += (double) a[i] * b[i];
                dot0 += (double) a0[i] * b0[i];
                na += (double) a[i] * a[i];
                na0 += (double) a0[i] * a0[i];
                nb += (double) b0[i] * b0[i];
            }
            worst_norm = std::max(worst_norm, std::fabs(std::sqrt(na) - std::sqrt(na0)) / std::sqrt(na0));
            worst_dot = std::max(worst_dot, std::fabs(dot - dot0) / (std::sqrt(na0) * std::sqrt(nb)));
            fwht(a);
            fwht(b);   // H . H = 1
            for (int i = 0; i < 256; ++i)
                worst_inv = std::max(worst_inv, (double) std::fabs(a[i] - a0[i]));
        }
        expect(worst_norm <= 1e-4, "the rotation keeps a row's length: the stages and the 1/16 are orthonormal");
        expect(worst_inv <= 1e-4, "the rotation is its own inverse, so the output un-rotation restores the row");
        expect(worst_dot <= 1e-4, "a rotated query and key keep their dot product: Q K^T is invariant under it");
        std::printf("  rotation: |Hx| vs |x| %.2g, H(Hx) vs x %.2g, (Hq).(Hk) vs q.k %.2g (relative)\n", worst_norm, worst_inv,
                    worst_dot);

        // the round trip the 4-bit caches make of K and V: rotate, quantize to q4_0 blocks, un-rotate. The error is the
        // format's own, spread over the row rather than concentrated in the blocks an outlier sits in.
        for (int round = 0; round < 100; ++round) {
            float row[256], rot[256];
            for (int i = 0; i < 256; ++i) row[i] = rot[i] = n(r);
            fwht(rot);
            float rt[256];
            for (int b = 0; b < 8; ++b) {
                const k::KvQ4Block blk = k::kv_q4_quantize_host(&rot[b * 32]);
                for (int j = 0; j < 32; ++j) rt[b * 32 + j] = k::kv_q4_dequant(element_code(blk, j), f16(blk.d));
            }
            fwht(rt);   // H (H V) = V: what the attention output's un-rotation gives
            double e2 = 0.0, x2 = 0.0;
            for (int i = 0; i < 256; ++i) e2 += (double) (rt[i] - row[i]) * (rt[i] - row[i]), x2 += (double) row[i] * row[i];
            worst_rt = std::max(worst_rt, std::sqrt(e2 / x2));
        }
        std::printf("  rotate, q4_0, un-rotate: relative L2 error %.3g over 100 random rows\n", worst_rt);
        expect(worst_rt <= 0.3, "a rotated-and-back 4-bit row is within the format's rounding of the row");
        {
            // one 64-valued dim and 255 zeros: unrotated, its block's step is the outlier over 8; rotated, every block's
            // step is the outlier over 128, which is why the caches rotate before 4 bits
            float row[256] = {0.0f}, rot[256];
            row[7] = 64.0f;
            for (int i = 0; i < 256; ++i) rot[i] = row[i];
            fwht(rot);
            float d_plain = 0.0f, d_rot = 0.0f;
            for (int b = 0; b < 8; ++b) {
                d_plain = std::max(d_plain, std::fabs(f16(k::kv_q4_quantize_host(&row[b * 32]).d)));
                d_rot = std::max(d_rot, std::fabs(f16(k::kv_q4_quantize_host(&rot[b * 32]).d)));
            }
            std::printf("  one 64-valued dim: the largest block step is %.3g unrotated, %.3g rotated\n", d_plain, d_rot);
            expect(d_rot * 8.0f <= d_plain, "the rotation spreads one outlier over every block, so no block's step is its eighth");
        }
    }

    // ---- the reconstruction bound over random rows, printed for the report
    {
        double worst = 0.0, worst_rel = 0.0;
        for (int round = 0; round < 100; ++round) {
            float row[256];
            for (float& v : row) v = nd(rng) * nd(rng);
            float amax = 0.0f;
            for (float v : row) amax = std::max(amax, std::fabs(v));
            for (int b = 0; b < 8; ++b) {
                const k::KvQ4Block blk = k::kv_q4_quantize_host(&row[b * 32]);
                const float d = f16(blk.d);
                for (int j = 0; j < 32; ++j) {
                    const double e = std::fabs(k::kv_q4_dequant(element_code(blk, j), d) - row[b * 32 + j]);
                    worst = std::max(worst, e);
                    if (row[b * 32 + j] != 0.0f) worst_rel = std::max(worst_rel, e / std::fabs(row[b * 32 + j]));
                    if (e > (double) std::fabs(d) + 1e-6) ++g_fail;
                }
                expect(std::fabs(d) <= amax / 8.0f * 1.001f + 1e-9f, "the step is at most an eighth of the row's largest |value|");
            }
        }
        std::printf("  worst reconstruction error %.6g (absolute), %.6g (relative)\n", worst, worst_rel);
    }

    std::printf("kv_q4_layout: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
